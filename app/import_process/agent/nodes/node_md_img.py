import os
import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.image_enrichment import summarize_images
from app.import_process.agent.image_storage import get_image_storage
from app.import_process.agent.markdown_image_rewriter import (
    compile_markdown_image_reference_pattern,
    replace_markdown_images,
)
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

"""
 1.校验并获取 本次所需要的数据对象 (校验md_path,与md_content
    :param state 
        对md_path 与md_content 进行校验
    :return: 返回 校验后的md_path 与图片的images对象，md_path_obj,md_content
 2.对上一步返回的对象进行处理，通过images目录获取到其下所有图片，并返回图片的上下文
    :param md_path_obj,md_content,images_path_obj
        从md文件中提取出images 遍历 判断images 中图片是否存在md文件中
    :return: 返回处理过的上下文[(图片名,图片地址,(上文，下文)]
 3.将图片上下文与图片名发给大模型 进行处理，返回{图片名：总结和描述}
    :param targets 图片的上下文  stem 图片名
    :return: 返回图片的总结
 4.将图片上传到miniIo 中
    :param summaries md_content, targets stem
    先将原桶内图片删除，再将需要更新的图片上传，同时更新md_content 中的图片地址
    :return summary {"图片名":(描述，内容)}
 5.将sction 中的文档切块保存到文件中进行备份
"""
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO 支持的图片格式(后缀不区分大小写）
    :param filename:
    :return:
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_validate_paths(state: ImportGraphState):
    md_path = state["md_path"]
    md_content = state["md_content"]
    #判断md_path 是否存在
    if not md_path:
        raise ValueError(f"[node_md_img]节点[step_1_validate_paths]中{md_path} 不存在")

    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"[node_md_img]节点[step_1_validate_paths]中{md_path_obj}文件不存在")
    # 判断 md_content 中是否有值
    # 如果没有值可能不是pdf 中转节点进来 ，中md_path下进行读取
    if not md_content:
        #没有值进行读取
        with open(md_path_obj, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()
            md_content = state["md_content"]
    images_path_obj = md_path_obj.parent / "images"
    if not images_path_obj.exists():
        logger.warning(f"[node_md_img] 节点[step_1_validate_paths] 中 {images_path_obj} 图片目录不存在，将优雅跳过图片描述与上传逻辑。")
        images_path_obj = None
    return md_path_obj, md_content, images_path_obj


def find_image_in_md_content(md_content, img_file, context_length: int = 100):
    # 正则表达式
    pattern = compile_markdown_image_reference_pattern(img_file)
    content = None  # 存储图片多处使用

    # 【安全改进】：使用 next() 获取迭代器的第一个元素，若为空则返回 None，避免 IndexError
    item = next(pattern.finditer(md_content), None)

    if item:
        start, end = item.span()
        # 截取上文
        pre_text = md_content[max(start - context_length, 0):end]
        post_text = md_content[end:min(end + context_length, len(md_content))]
        # 上下文
        content = (pre_text, post_text)

    if content:
        logger.info(f"图片：{img_file} 上下文:{content[0]}")
        return content

    return None


def step_2_scan_images(md_content, images_path_obj: Path):
    #创建一个目标集合
    targets = []
    if not images_path_obj or not images_path_obj.exists():
        logger.info("[step_2_scan_images] 图片目录不存在或无图片，扫描图片结果为空")
        return targets
    #返回图片对应的上下文
    #遍历 图片地址下的图片
    for img_file in os.listdir(images_path_obj):
        #遍历每一个文件的名字
        #检查图片是否可用
        if not is_supported_image(img_file):
            logger.warning(f"当前文件：{img_file}，不是图片格式，无需处理")
            continue
        #是图片
        content_data = find_image_in_md_content(md_content, img_file)
        if not content_data:
            logger.warning(f"图片：{img_file}没有md内容使用！上下文为空！")
            continue
        targets.append((img_file,str(images_path_obj/img_file),content_data))
    return targets


def step_3_generate_img_summaries(targets, name):
    """
    将图片发送给大模型进行总结（多线程并发识别，带实时日志与耗时打印）
    :param targets: 图片文件上下文列表 (img_file, image_path, context)
    :param name: 文件名
    :return: summaries 字典 {img_file: summary}
    """
    if not targets:
        logger.info("[step_3_generate_img_summaries] targets 为空，跳过图片识别")
        return {}
    return summarize_images(targets, name)


def step_4_upload_images_and_replace_md(summarise, targets, md_content, stem):
    """
        上传图片并替换原md 中的图片和描述
    :param summarise: 图片名:描述
    :param targets: (图片名，原地址，(上下文))
    :param md_content: 原md 内容
    :param stem: 文件名
    :return: 新md
    """
    if not targets:
        logger.info("[step_4_upload_images_and_replace_md] targets为空，无需上传图片至MinIO")
        return md_content

    image_targets = [(image_file, image_path) for image_file, image_path, _ in targets]
    image_url = get_image_storage().replace_images_for_document(stem, image_targets)
    image_infos = {
        image_file: (summarise.get(image_file, ""), url)
        for image_file, url in image_url.items()
    }
    logger.info(f"图片处理的汇总结果：{image_infos}")
    if image_infos:
        md_content = replace_markdown_images(md_content, image_infos)
        logger.info(f"已完成md_content替换，新的md_content为{md_content}")
    return md_content


def step_5_replace_md_content(new_md_content, md_path_obj):
    """
        将旧 地址替换为新地址
    :param new_md_content: md新地址
    :param md_path_obj: #旧md 地址
    :return: #新地址
    """
    #获取到路径
    new_md_path = os.path.splitext(md_path_obj)[0]+"_new.md"
    #将新的内容写入到新文件中
    with open(new_md_path,"w",encoding="utf-8") as f:
        f.write(new_md_content)
    #返回新文件的路径
    logger.info(f"已经完成了新内容的写入，新地址为：{new_md_path}")
    return new_md_path

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {node_name}")
    add_running_task(state['task_id'],node_name)

#     1.
#     校验并获取
#     本次所需要的数据对象(校验md_path, 与md_content
#     : param state
#     对md_path  与md_content    进行校验
#     return: 返回校验后的md_path 与图片的images对象，md_path_obj, md_content
    md_path_obj, md_content, images_path_obj = step_1_validate_paths(state=state)


#     2.
#     对上一步返回的对象进行处理，通过images目录获取到其下所有图片，并返回图片的上下文
#     :param
#     md_path_obj, md_content, images_path_obj
#     从md文件中提取出images
#     遍历
#     判断images
#     中图片是否存在md文件中
#
# :return: 返回处理过的上下文[(图片名, 图片地址, (上文，下文)]

    targets = step_2_scan_images(md_content,images_path_obj)
    #将上下文和图片发给大模型 返回类型为 summarise {"图片名":(上下文)}
    summarise = step_3_generate_img_summaries(targets,md_path_obj.name)

    #将图片上传至MiniIO 中进行存储
    new_md_content = step_4_upload_images_and_replace_md(summarise=summarise,targets=targets,md_content=md_content,stem=md_path_obj.stem)
    #将内容进行替换 返回一个新md 地址
    new_md_path = step_5_replace_md_content(new_md_content,md_path_obj)
    state['md_path'] = new_md_path
    state['md_content'] = new_md_content
    logger.info(f">>> [Stub] 结束节点: {node_name}")
    add_done_task(state['task_id'],node_name)
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
