import base64
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载 .env 环境变量
load_dotenv()

# 2. 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.lm.lm_utils import get_vlm_client
from app.core.logger import logger


def test_vlm_vision():
    logger.info("=== 开始测试 VLM 视觉大模型连接与识别能力 ===")

    # 1. 获取独立的 VLM 客户端
    try:
        vlm_client = get_vlm_client()
        logger.info(f"[成功] 成功获取 VLM 客户端，实例模型：{vlm_client.model_name}")
    except Exception as e:
        logger.error(f"[失败] 获取 VLM 客户端失败，请检查 .env 配置：{e}")
        return

    # 2. 寻找本地 output 目录下的真实图片进行测试；若没有则使用红点测试图
    test_image_path = None
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break

    if test_image_path and os.path.exists(test_image_path):
        logger.info(f"[图片] 找到本地测试图片进行测试：{test_image_path}")
        with open(test_image_path, "rb") as f:
            test_image_base64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        logger.info("[提示] 未找到本地测试图片，使用基础像素图测试接口连通性...")
        # 1x1 简易测试图片 Base64
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    # 3. 构造 LangChain 多模态 Message 结构
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{test_image_base64}"
                    },
                },
                {
                    "type": "text",
                    "text": "请简要描述一下这张图片中的主要内容，并用一句话总结。"
                }
            ]
        }
    ]

    # 4. 发送请求并打印响应结果
    try:
        logger.info("[发送] 正在向 VLM 视觉模型发送图文多模态请求...")
        response = vlm_client.invoke(messages)
        print("\n" + "=" * 50)
        print("[成功] VLM 视觉大模型响应成功！解析描述结果如下：")
        print(response.content.strip())
        print("=" * 50 + "\n")
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"[失败] VLM 视觉模型请求失败或超时！异常信息：{e}")
        print("建议：检查 .env 中 VLM_BASE_URL、VLM_API_KEY 与 VLM_TIMEOUT 配置")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    test_vlm_vision()