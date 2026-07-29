# 定义状态图和编译对象
# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库


# 初始化环境变量：必须在配置读取前执行，确保后续节点能获取到环境变量中的配置信息
load_dotenv()

#1.初始化langgraph
work_flow = StateGraph(ImportGraphState)
#2.注册节点
work_flow.add_node("node_entry", node_entry)
work_flow.add_node("node_pdf_to_md", node_pdf_to_md)
work_flow.add_node("node_md_img", node_md_img)
work_flow.add_node("node_document_split", node_document_split)
work_flow.add_node("node_item_name_recognition", node_item_name_recognition)
work_flow.add_node("node_bge_embedding", node_bge_embedding)
work_flow.add_node("node_import_milvus", node_import_milvus)


#3.确定起始节点
work_flow.set_entry_point("node_entry")

#4.定义条件路由 跳转
def route_node(state:ImportGraphState)->str:
    """
    定义路由跳转条件
    :param state:
    :return: 跳转的路由
    """
    if state["is_md_read_enabled"]:
        return "node_md_img" #直接去md提取图片节点
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md" #去pdf 转md节点
    else:
        return END #退出所有节点
#5.确定各节点边
work_flow.add_conditional_edges("node_entry", route_node,{
    "node_md_img":"node_md_img",
    "node_pdf_to_md":"node_pdf_to_md",
    END : END
})

work_flow.add_edge("node_pdf_to_md","node_md_img") #pdf 转md 后跳转到提取md图片下载
work_flow.add_edge("node_md_img","node_document_split") #md 提取完整后对文档进行切块
work_flow.add_edge("node_document_split","node_item_name_recognition") #切块后提取文档的标题名称
work_flow.add_edge("node_item_name_recognition","node_bge_embedding") #将文档内容以及标题进行向量化
work_flow.add_edge("node_bge_embedding","node_import_milvus") #将向量化后的数据存入向量数据库中
work_flow.add_edge("node_import_milvus",END) #结束


#6.编译得到图节点对象
kb_work_app = work_flow.compile()




