import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.lm.reranker_utils import get_reranker_model

print("正在加载 Reranker 模型...")
model = get_reranker_model()

print("开始测试简单的 compute_score...")
scores = model.compute_score([["什么是 RRF？", "RRF 是一种融合算法"]], normalize=True)
print("计算成功，得分:", scores)