import argparse

from app.vector_store.milvus_adapter import MilvusVectorStore


CONFIRMATION = "DROP_AND_RECREATE_MILVUS_VECTOR_COLLECTIONS"


def load_application_services():
    from app.application_services import create_application_services_from_env

    return create_application_services_from_env()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    if not args.user_id or args.confirm != CONFIRMATION:
        print("确认短语不匹配，未执行任何删除")
        return 2
    services = load_application_services()
    services.initialize_database_only()
    snapshot = services.settings.get_snapshot(args.user_id)
    if snapshot.vector_store_type != "milvus" or snapshot.milvus is None:
        print("该用户当前没有有效的 Milvus 配置")
        return 2
    MilvusVectorStore(snapshot.milvus).rebuild_collections()
    print("Milvus 两个 collection 已按新 schema 重建")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
