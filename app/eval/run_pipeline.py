"""
离线跑查询图，收集 RAGAS 样本。

用法（项目根目录）:
  python -m app.eval.run_pipeline
  python -m app.eval.run_pipeline --golden app/eval/golden_set.jsonl --out app/eval/reports/pipeline_outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any
from langgraph.types import Command
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import get_task_result

# app/eval/ 目录（本文件所在目录），与 CWD 无关
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden_set.jsonl"
DEFAULT_OUT = EVAL_DIR / "reports" / "pipeline_outputs.jsonl"


def resolve_path(path: str | Path) -> Path:
    """
    解析路径，不依赖当前工作目录：
    - 绝对路径：原样返回
    - 相对路径：相对 app/eval/ 解析
    - 兼容从项目根写的 app/eval/...
    """
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "app" and parts[1] == "eval":
        p = Path(*parts[2:]) if len(parts) > 2 else Path()
    return (EVAL_DIR / p).resolve()


def _docs_to_contexts(reranked_docs: list[dict] | None) -> list[str]:
    if not reranked_docs:
        return []
    contexts: list[str] = []
    for doc in reranked_docs:
        text = (doc or {}).get("text") or ""
        text = text.strip()
        if text:
            contexts.append(text)
    return contexts


def run_one(question: str, *, max_contexts: int = 10) -> dict[str, Any]:
    session_id = f"eval-{uuid.uuid4().hex[:12]}"
    request_id = str(uuid.uuid4())

    state = create_query_default_state(
        session_id=session_id,
        request_id=request_id,
        original_query=question,
        is_stream=False,
    )
    config = {"configurable": {"thread_id": session_id}}

    rewritten_query = question
    item_names: list[str] = []
    contexts: list[str] = []
    answer = ""
    status = "ok"
    error = ""

    try:
        # 第一阶段：正常流式运行图
        for event in query_app.stream(state, config, stream_mode="updates"):
            # 兼容判断 interrupt
            if isinstance(event, dict) and "__interrupt__" in event:
                status = "awaiting_confirmation"
                break
            if not isinstance(event, dict):
                continue

            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    status = "awaiting_confirmation"
                    break
                if not isinstance(update, dict):
                    continue

                if update.get("rewritten_query"):
                    rewritten_query = update["rewritten_query"]
                if update.get("item_names") is not None:
                    item_names = list(update.get("item_names") or [])
                if update.get("reranked_docs"):
                    contexts = _docs_to_contexts(update["reranked_docs"])[:max_contexts]
                if update.get("answer"):
                    answer = update["answer"]

        # -------------------------------------------------------------
        # 🔑 关键修复：如果中断等待商品确认，评估模式下自动选择匹配到的候选商品继续执行
        # -------------------------------------------------------------
        if status == "awaiting_confirmation":
            # 恢复图执行（传入 resume 信号，选择默认候选商品）
            # 假设你的节点在 resume 时接受选中的商品名称或第一个候选
            resume_input = Command(resume={"candidate_id": item_names[0] if item_names else "华为擎云B530"})

            # 重新流式执行剩余节点
            for event in query_app.stream(resume_input, config, stream_mode="updates"):
                if not isinstance(event, dict):
                    continue
                for node_name, update in event.items():
                    if not isinstance(update, dict):
                        continue
                    if update.get("reranked_docs"):
                        contexts = _docs_to_contexts(update["reranked_docs"])[:max_contexts]
                    if update.get("answer"):
                        answer = update["answer"]

            # 恢复成功后将状态改回 ok
            status = "ok"

        # 获取最终答案
        task_answer = get_task_result(request_id, "answer", default="")
        if task_answer:
            answer = task_answer

        if not answer and status == "ok":
            status = "empty_answer"
            error = "answer is empty after pipeline"

    except Exception as e:
        status = "error"
        error = str(e)
        logger.exception(f"[eval] run_one failed: {question}")

    return {
        "user_input": rewritten_query or question,
        "retrieved_contexts": contexts,
        "response": answer or "",
        "meta": {
            "original_query": question,
            "rewritten_query": rewritten_query,
            "item_names": item_names,
            "session_id": session_id,
            "request_id": request_id,
            "status": status,
            "error": error,
            "num_contexts": len(contexts),
        },
    }


def build_dataset(golden_path: Path, out_path: Path, max_contexts: int = 10) -> list[dict]:
    rows: list[dict] = []
    golden_lines = golden_path.read_text(encoding="utf-8").splitlines()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 打开文件句柄，每处理完一条立刻写入磁盘并 flush
    with out_path.open("w", encoding="utf-8") as f:
        for i, line in enumerate(golden_lines, start=1):
            line = line.strip()
            if not line:
                continue
            gold = json.loads(line)
            question = gold["question"]
            logger.info(f"[eval] ({i}) running: {question}")

            try:
                sample = run_one(question, max_contexts=max_contexts)
            except Exception as e:
                logger.error(f"[eval] ({i}) 样本运行严重崩溃: {e}")
                continue

            sample["reference"] = gold.get("ground_truth", "")
            sample["meta"]["expected_item_names"] = gold.get("item_names", [])
            rows.append(sample)

            # 边跑边存
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

            logger.info(
                f"[eval] ({i}) status={sample['meta']['status']} "
                f"contexts={sample['meta']['num_contexts']} "
                f"answer_len={len(sample.get('response') or '')}"
            )

    logger.info(f"[eval] wrote {len(rows)} samples -> {out_path}")
    return rows

def main():
    parser = argparse.ArgumentParser(description="Run query pipeline for RAGAS dataset")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="金标 jsonl 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="pipeline 输出 jsonl 路径")
    parser.add_argument("--max-contexts", type=int, default=10)
    args = parser.parse_args()

    golden_path = resolve_path(args.golden)
    out_path = resolve_path(args.out)
    logger.info(f"[eval] golden={golden_path}")
    logger.info(f"[eval] out={out_path}")
    if not golden_path.exists():
        raise FileNotFoundError(f"金标文件不存在: {golden_path}")

    build_dataset(golden_path, out_path, max_contexts=args.max_contexts)


if __name__ == "__main__":
    main()
