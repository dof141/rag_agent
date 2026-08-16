"""
对 pipeline 产出做 RAGAS 评估 (适配 Ragas 最新版本规范)

用法:
  python -m app.eval.run_ragas
  python -m app.eval.run_ragas --input app/eval/reports/pipeline_outputs.jsonl --out app/eval/reports/ragas_scores.csv
"""
from __future__ import annotations

# =====================================================================
# 🛠️ 【核心修复】拦截 Ragas 底层硬编码的旧版 VertexAI 模块崩溃问题
# 必须放置在任何 ragas 相关的 import 语句之前！
# =====================================================================
import sys
import types

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex
# =====================================================================

import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
)

from app.core.logger import logger
from app.eval.run_pipeline import summarize_samples
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import get_bge_m3_ef

# answer_relevancy 可选导入
try:
    from ragas.metrics import answer_relevancy
    HAS_ANSWER_RELEVANCY = True
except Exception:
    HAS_ANSWER_RELEVANCY = False

# app/eval/ 目录（本文件所在目录），与 CWD 无关
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EVAL_DIR / "reports" / "pipeline_outputs.jsonl"
DEFAULT_OUT = EVAL_DIR / "reports" / "ragas_scores.csv"


def resolve_path(path: str | Path) -> Path:
    """绝对路径原样返回；相对路径相对 app/eval/ 解析"""
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "app" and parts[1] == "eval":
        p = Path(*parts[2:]) if len(parts) > 2 else Path()
    return (EVAL_DIR / p).resolve()


def load_samples(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def to_ragas_rows(samples: list[dict], *, scorable_only: bool = False) -> list[dict]:
    """
    转换为符合 Ragas 最新规范的标准 Dataset 数据字典：
    - user_input: 用户输入问题
    - retrieved_contexts: 检索到的上下文列表 (list[str])
    - response: 大模型生成的回答
    - reference: 真实标准答案 (Ground Truth)
    """
    rows = []
    for s in samples:
        meta = s.get("meta") or {}
        status = meta.get("status", "ok")
        user_input = s.get("user_input") or meta.get("original_query") or ""
        contexts = s.get("retrieved_contexts") or []
        response = s.get("response") or ""
        reference = s.get("reference") or ""

        if scorable_only and (
            status not in {"success", "ok", "final"}
            or not response
            or not contexts
        ):
            logger.warning(
                f"[ragas] explicitly excluded unscorable sample "
                f"status={status} q={user_input[:40]}"
            )
            continue

        rows.append(
            {
                "user_input": user_input,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": reference,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline outputs with RAGAS")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="pipeline 输出 jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="分数 csv 输出路径")
    parser.add_argument("--model", default=None, help="裁判模型名")
    parser.add_argument("--with-answer-relevancy", action="store_true", help="开启 answer_relevancy（需 embeddings）")
    parser.add_argument(
        "--scorable-only",
        action="store_true",
        help="显式排除失败或空依据样本；默认保留全部样本",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    out_path = resolve_path(args.out)
    logger.info(f"[ragas] input={input_path}")
    logger.info(f"[ragas] out={out_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"pipeline 输出不存在，请先跑 run_pipeline: {input_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_samples(input_path)
    pipeline_summary = summarize_samples(raw)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({"pipeline": pipeline_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[ragas] pipeline summary={pipeline_summary}")

    rows = to_ragas_rows(raw, scorable_only=args.scorable_only)
    if not rows:
        raise SystemExit("没有可评估样本，请先检查 pipeline 输出")

    dataset = Dataset.from_list(rows)
    logger.info(f"[ragas] evaluating {len(rows)} samples")

    # 1. 裁判 LLM 包装（适配最新 Ragas 接口）
    judge_llm = LangchainLLMWrapper(get_llm_client(model=args.model))

    # 2. 核心评估指标列表
    metrics = [
        faithfulness,       # 答案忠实度
        context_recall,     # 检索召回率
        context_precision,  # 检索精确度
    ]

    # 3. 嵌入模型（如果开启了 answer_relevancy）
    judge_embeddings = None
    if args.with_answer_relevancy and HAS_ANSWER_RELEVANCY:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        base_embed = get_bge_m3_ef()
        judge_embeddings = LangchainEmbeddingsWrapper(base_embed)
        metrics.append(answer_relevancy)

    # 4. 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )

    print(result)

    # 保存评估报告
    df = result.to_pandas()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[ragas] scores saved -> {out_path}")

    # 保存 Summary 汇总 JSON
    try:
        metric_summary = dict(result)
    except Exception:
        metric_summary = {"result": str(result)}
    summary = {
        "pipeline": pipeline_summary,
        "ragas": metric_summary,
        "evaluated_samples": len(rows),
        "scorable_only": args.scorable_only,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[ragas] summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
