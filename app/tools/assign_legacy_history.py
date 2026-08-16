import argparse


CONFIRMATION = "ASSIGN_LEGACY_HISTORY"


def assign_legacy_history(collection, *, user_id: str, confirm: str) -> int:
    if not user_id:
        raise ValueError("user_id is required")
    if confirm != CONFIRMATION:
        return 0
    result = collection.update_many(
        {"user_id": {"$exists": False}},
        {"$set": {"user_id": user_id}},
    )
    return result.modified_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Assign legacy history records without user_id to one user."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args(argv)

    if args.confirm != CONFIRMATION:
        print(f"Refusing migration: --confirm must equal {CONFIRMATION}")
        return 2

    from app.clients.mongo_history_utils import get_history_mongo_tool

    count = assign_legacy_history(
        get_history_mongo_tool().chat_message,
        user_id=args.user_id,
        confirm=args.confirm,
    )
    print(f"Assigned {count} legacy history records to user {args.user_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
