import json
import os.path
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy.polys.domains import field

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task

"""
#1 校验参数是否存在 
    :param md_content, file_tile #标题
    :return md_content, file_tile
#2.对文档进行切割 按照标题 进行切割
    :param md_content, file_tile
    :return [{content,title,file_title}] 返回 标题内容，标题，和文件名   
#3.对文档进行精切，判断是否大于或者小于某长度
    :param section #存储文档的集合 包含文档名称，文档内容，文档标题
    :return finally #新的文档集合
"""
# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500
#切割符号
SEPARATORS=['\n\n',"\n",'。',"，","!",":"]

def step_1_get_inputs(state:ImportGraphState):
    """
    对文档内容和file_title 进行校验
    :param state:
    :return:
    """
    md_content = state["md_content"]
    if not md_content:
        logger.error(f"节点[node_document_split] 中[step_1_get_inputs]方法 md_content:{md_content}不能为空")
        raise ValueError(f"节点[node_document_split] 中[step_1_get_inputs]方法 md_content:{md_content}不能为空")
    #对文件名称进行判断
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n") #将换行替换
    file_title = state.get("file_title","default_title") #获取到 file_title 如果不存在则赋默认值
    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    """
        对文档进行 语义切割
    :param md_content:
    :param file_title:
    :return: sections,title_count,len(lines_list) 切块后的文档， 切块数量，切割数量
    """
    #定义正则表达式
    #^\s* 表示空格 多个
    # #{1,6} 表示 匹配 # 1个到6个 标题名
    # .+ .任意字符串 + 1 -> n [空格]###[空格] 标题描述

    title_pattern = r"^\s*#{1,6}\s+.+"
    lines_list = md_content.split("\n") #对文档进行切割

    current_title = ""
    current_lines = [] #当前标题行
    title_count = 0
    is_code_block = False #判断是否为代码块行
    # 最终存储的列表 sections=[]
    sections = []

    for line in lines_list:
        line_strip = line.strip()
        #判断当前行是否为代码块行
        if line_strip.startswith("~~~") or line_strip.startswith("```"):
            #当前行为代码块行
            is_code_block = not is_code_block #取反
            #代码块 则直接将内容追加进 lines_content
            current_lines.append(line_strip)
            continue
        #不是代码块，判断是否为标题
        is_title = not is_code_block and re.match(title_pattern, line_strip) #正则匹配判断是否为标题 同时判断是否为代码块中内容
        if is_title:
            #将内容加入到最终返回结果中
            #判断是否为第一次 第一次中 current_title 没有值
            if current_title: #当前有标题，证明已经到下一个标题
                sections.append({
                  "title": current_title,
                  "content": "\n".join(current_lines),
                  "file_title": file_title,
                })
            current_title = line_strip
            current_lines = [current_title]
            title_count += 1 #标题数量+1
        else:
            #将行内容加入到行内容列表中
            current_lines.append(line_strip)
        # 将内容加入到最终返回结果中
    # 判断是否为第一次 第一次中 current_title 没有值
    if current_title:  # 当前有标题，证明已经到下一个标题 将最后一个标题中的内容存储
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })
    logger.info(f"已经完成chunks的语义粗切！识别chunk数量：{title_count},切片内容:{sections}")
    return sections,title_count,len(lines_list)


def split_long_section(section, max_length):
    """
    对过长的文档进行再次切割
    :param section:
    :param max_length:
    :return:
    """
    #对传入的集合中内容长度进行判断 是否超长
    content = section.get("content")
    if len(content) <= max_length: #没有超出长度不进行处理，直接返回
        logger.info(f"[split_long_section]:{content},当前chunk长度小于等于{max_length}，不做二次切割")
        return [section]
    #使用langchain中的 切割函数
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,#重叠长度
        separators=SEPARATORS #切割符号
    )
    sub_sections = []
    for index,chunk in enumerate(splitter.split_text(content),start=1):
        text = chunk.strip() #切片内容
        title = f"{section.get('title')}_{index}"
        part_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": part_title,
            "part": part,
        })
    return sub_sections


def merge_short_sections(finally_sections, min_length):
    """
    对小于的集合进行合并
    :param finally_sections:
    :param min_length:
    :return:
    """
    merged_sections = [] #存储合并后的结果
    pre_section=None #前一次的content
    for section in finally_sections:
        if not pre_section: #第一次进入 给pre_content赋值
            pre_section = section
            continue
        #对pre_content 进行判断 是否小于最小的长度
        is_current_short = len(pre_section.get("content")) <min_length #小于
        is_same_parent_title =section.get('parent_title')and( section.get("parent_title") == pre_section.get("parent_title"))
        #如果是同一个父标题 并且上一次的长度小于最小长度 则进行合并
        if is_current_short and is_same_parent_title:
            #同一个父标题 合并
            current_content = section.get("content")
            pre_section['content'] +="\n\n" + current_content
            pre_section['part'] = section.get("part")
        else:
            #不是同一个父标题
            merged_sections.append(section)
            pre_section = section
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections

def step_3_refine_chunks(sections, max_length, min_length):
    """
    对文档进行精细切割
    :param sections: 传入的文档结合
    :param DEFAULT_MAX_CONTENT_LENGTH: 单块最大长度
    :param MIN_CONTENT_LENGTH: 单块最新长度
    :return: 返回新的集合
    """
    finally_sections = []
    for section in sections:
        #大于了最大的默认长度，进行再次切割
        sub_section = split_long_section(section,max_length)
        finally_sections.extend(sub_section) #追加
    #小于则再合并
    finally_sections= merge_short_sections(finally_sections,min_length)
    for section in finally_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_tile') or section.get("title")
    return finally_sections


def step_6_backup(state, finally_sections):
    """
    保存备份文件
    :param state:
    :param finally_sections:
    :return:
    """
    local_dir = state['local_dir']
    backup_file_path = os.path.join(local_dir,"chunks.json")
    with open(backup_file_path,"w",encoding="utf-8") as f:
        json.dump(
            finally_sections,
            f,
            ensure_ascii=False,
            indent=4
        )
    logger.info(f"已经将内容，进行备份到：{backup_file_path}中")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)
    try:

        #对md_content 和file_title 进行校验
        md_content, file_title = step_1_get_inputs(state)
        #对文档进行粗切 使用 标题 进行语义进行切割
        sections,title_count,lines_count=step_2_split_by_title(md_content, file_title)
        #对标题数量进行判断，如果数量为零 则设置默认标题
        if title_count == 0:
            sections.append({
                "title": "没有标题",
                "content":md_content,
                "file_title":file_title,
            })
        #对文档内容进行和并判断，如果大于指定长度进行切割，小于则进行合并 返回新的sections[{标题,内容,文件名}]
        finally_sections = step_3_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        #将sections 进行保存
        state['chunks'] = finally_sections
        step_6_backup(state,finally_sections)


    except Exception as e:
        logger.error(f"[node_document_split]节点出现异常")
    finally:
        # 结束当前节点信息，用于任务监控和日志溯源
        logger.info(f">>> 结束执行核心节点：【文档切分】{node_name}")
        # 将当前节点加入运行中任务，更新全局任务状态
        add_running_task(state["task_id"], node_name)
        return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")