from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))


def main():
    from app.core.logger import logger
    from app.import_process.agent.main_graph import build_import_graph
    from app.import_process.runtime import ImportRuntime
    from app.vector_store.interface import VectorImportResult

    class FakeEmbedding:
        def embed_documents(self, texts):
            return {"dense": [[1.0, 0.0] for _ in texts]}

    class FakeVectorStore:
        def import_document(self, document):
            return VectorImportResult(item_count=1, chunk_count=len(document.chunks))

    logger.info("===== 开始图结构检查 =====")

    runtime = ImportRuntime(embedding=FakeEmbedding(), vector_store=FakeVectorStore())
    graph = build_import_graph(runtime)
    logger.info("图结构")
    try:
        graph.get_graph().print_ascii()
    except Exception as exc:
        logger.info("图结构打印跳过: %s", exc)
    logger.info("===== 图结构检查结束 =====")


if __name__ == "__main__":
    main()
