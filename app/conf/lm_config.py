# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义 LLM 与 VLM 服务配置
@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    lv_model: str
    llm_model: str
    llm_temperature: float
    # VLM 视觉模型独立配置
    vlm_base_url: str
    vlm_api_key: str
    vlm_timeout: float
    item_name_timeout: float

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE") or 0.1),
    # 优先读取 VLM_* 环境变量，若未配置则优雅降级为通用 OPENAI 配置
    vlm_base_url=os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    vlm_api_key=os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    vlm_timeout=float(os.getenv("VLM_TIMEOUT") or 60.0),
    item_name_timeout=float(os.getenv("ITEM_NAME_TIMEOUT_SECONDS") or 12.0),
)
