import os
import shutil
import sys
import time
import zipfile
from pathlib import Path


import requests
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.conf.mineru_config import mineru_config
from app.utils.task_utils import add_running_task, add_done_task


def step_1_validate_paths(state):
    #对传入地址已经文件进行判断
    #拿到文件路径和存储路径
    log_prefix = "[step_1_validate_paths] "
    local_dir = state["local_dir"]
    pdf_path = state["pdf_path"]
    #进行路径非空校验
    if not pdf_path:
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：pdf_path，当前值：{repr(pdf_path)}")
    if not local_dir:
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：pdf_path，当前值：{repr(local_dir)}")
    #对文件提取path对象
    local_dir_path = Path(local_dir)
    # 确保输出目录存在，不存在则递归创建
    if not local_dir_path.is_dir():
        logger.info(f"{log_prefix}输出目录不存在，自动创建：{local_dir_path.absolute()}")
        local_dir_path.mkdir(parents=True, exist_ok=True)

    #对pdf 提取path对象 统一处理路径
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix}指定路径非文件（是目录），绝对路径：{pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix}PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")
    return pdf_path_obj, local_dir_path

def step_2_upload_and_poll(pdf_path_obj):
    """
    将文件上传 并且获取到解压文件下载的地址
    :param pdf_path_obj:
    :return:
    """
    token = mineru_config.api_key
    url = mineru_config.base_url+"/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    #将请求打进服务 使用post请求 如果返回的stute 不是200，或者code 不为0 则抛出异常
    response = requests.post(url=url,headers=header,json=data)
    # logger.info(f"完整响应: {response.json()}")  # 添加这一行
    if response.status_code != 200 or response.json()['code'] !=0:
        logger.error(f"[step_2_upload_and_poll]请求解析错误，请检查文件路径是否正确")
        raise RuntimeError(f"[step_2_upload_and_poll]请求解析错误，请检查文件路径是否正确")
    #得到上传文件id 与处理id
    uploaded_url = response.json()['data']['file_urls'][0]
    batch_id = response.json()['data']['batch_id'] #处理id

    #2.将文件上传到对应解析地址
    #使用put 请求
    #获取session 将虚拟地址关闭
    http_session = requests.session()
    http_session.trust_env = False #关闭代理
    try:
        file_data =None
        with open(pdf_path_obj,'rb') as f:
            file_data = f.read()
        if not file_data:
            logger.error(f"[step_2_upload_and_poll] file_content 无法读取，请检查文件路径是否正确 当前值{file_data}")
        uploaded_response = http_session.put(uploaded_url,data=file_data)
        if uploaded_response.status_code !=200:
            logger.error(f"[step_2_upload_and_poll]上传文件到minerU错误，请检查文件路径是否正确 1")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU错误，请检查文件路径是否正确2")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll] 上传文件到minerU失败，原始异常: {e}")
        raise RuntimeError(f"上传文件到minerU失败: {e}") from e
    finally:
        http_session.close() #关闭连接

    #3.轮询获取解析结果 限制10 分钟内
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600 #1s -> 1页pdf
    poll_inerval = 3 #间隔时间
    start_time = time.time()
    while True:
        #判断请求是否超过了最大的时间
        if time.time() - start_time >=timeout_seconds:
            raise TimeoutError(f"[任务轮询] 超时！任务处理超{int(timeout_seconds)}秒，batch_id：{batch_id}")

        #向指定地址获取解析结果
        res = requests.get(url,headers=header)
        #解析结果判断 和 zip url 地址获取
        if res.status_code != 200:
            #如果状态为500 到600 之间重试一次
            if 500<= res.status_code <600:
                time.sleep(poll_inerval)
                continue
            logger.error(f"[step_2_upload_and_poll]请求minerU接口失败，返回接口码{res.status_code}")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU错误，请检查文件路径是否正确5")
        #获取到本次的结果
        json_data = res.json()
        # logger.info(f"完整响应: {json_data}")  # 添加这一行
        if json_data['code'] != 0:
            raise RuntimeError(f"[任务轮询] API业务错误，返回数据：{json_data}")

        #判断解析状态
        extract_result_list = json_data['data']['extract_result']
        if not extract_result_list:
            time.sleep(poll_inerval)
            continue

        # 取第一个文件的结果（因为只上传了一个文件）
        extract_result = extract_result_list[0]
        if extract_result.get('state') == 'done':
            zip_url = extract_result.get('full_zip_url')
            if zip_url:
                logger.info(f"[step_2_upload_and_poll]已完成pdf解析耗时{time.time()-start_time}")
                return zip_url
            else:
                raise RuntimeError("解析成功但未返回 zip_url")
        else:
            # 状态为 waiting-file 或 processing，继续等待
            time.sleep(poll_inerval)



def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    #获取zip url地址下的zip 文件
    res  = requests.get(zip_url)
    if res.status_code !=200:
        logger.error(f"[step_3_download_and_extract]请求{zip_url}接口失败，返回接口码{res.status_code}")
        raise (f"[step_3_download_and_extract]请求{zip_url}接口失败，返回接口码{res.status_code}")
    #将zip文件保存
    zip_save_path =local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path,'wb') as f:
        f.write(res.content)
    logger.info(f"[step_3_download_and_extract] 下载zip 文件成功,保存地址{zip_save_path}")
    #对目标文件夹进行判断 是否已经存在 存在则删除
    extract_target_url = local_dir_obj / stem
    if extract_target_url.exists():
        #递归删除 会删除掉文件本身
        shutil.rmtree(extract_target_url)
    #创建
    extract_target_url.mkdir(parents=True,exist_ok=True)
    #使用zipFile 进行解压
    with zipfile.ZipFile(zip_save_path,'r') as zip_file_object:
        zip_file_object.extractall(extract_target_url)
    #判断解压名是否为原文件名
    md_file_list = list(extract_target_url.rglob("*.md"))
    #判断是否为 有最终md文件
    target_md_url=None
    #检查是否有原文件的md文件
    for md_file in md_file_list:
        if md_file.name == stem+".md":
            target_md_url = md_file
            break
    if not target_md_url:
        for md_file in md_file_list:
            if md_file.name.lower() =="full.md":
                target_md_url = md_file
                break
    #最终没有则获取第一个
    if not target_md_url:
        target_md_url = md_file_list[0]

    if target_md_url.stem != stem:
        target_md_url = target_md_url.rename(target_md_url.with_name(f"{stem}.md"))

    #返回最终路径
    finall_md_file_path = str(target_md_url.resolve())
    logger.info(f"[step_3_download_and_extract] 文件解压完成 最终存储路径{finall_md_file_path}")
    return finall_md_file_path





def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
        1. 进入的日志和任务状态的配置
        2. 进行参数校验 （local_dir -》 给与默认值 | local_file_path完成字面意思的校验 -》 深入校验校验的文件是否真的存在）
        3. 调用minerU进行pdf的解析（local_file_path）返回一个下载文件的地址 xx.zip url地址
        4. 下载zip包，并且解析和提取 （local_dir）
        5. 把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
        6. 结束的日志和任务状态的配置
        容错率处理！！ try异常处理
    """
    # 1. 进入的日志和任务状态的配置
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"], function_name)


    # 2 对文件进行解析处理
    try:
        # 分步骤 第一步校验目标文件夹和文件是否存在 同时返回文件目录和 文件名
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state=state)
        # 第二步，使用 mineru 进行对文件解析，并且得到 返回后的 md 文件下载地址
        zip_url = step_2_upload_and_poll(pdf_path_obj=pdf_path_obj)
        # 第三步，下载返回的 url 中的 zip 文件解压 并得到最终的 md 地址，将地址放入 state 中
        md_path = step_3_download_and_extract(zip_url=zip_url, local_dir_obj=local_dir_obj, stem=pdf_path_obj.stem)

        # 返回 state
        state['md_path'] = md_path
        with open(md_path, 'r', encoding="utf-8") as f:
            state['md_content'] = f.read()

        logger.info(f">>> [{function_name}] 结束执行了！现在的状态为：{state}")
        add_done_task(state["task_id"], function_name)
        return state

    except Exception as e:
        logger.error(f"[{function_name}] 使用 minerU 解析出现异常 异常信息：{e}")
        raise RuntimeError(f"MinerU 解析出现异常: {e}")



if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")