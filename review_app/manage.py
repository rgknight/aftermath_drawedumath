"""Run review-app chores against Firestore from your machine.

Reads GCP_PROJECT (and optionally FIRESTORE_DATABASE) from .env.deploy at the repo root.

    uv run python -m review_app.manage seed                 # load the subset (empty database only)
    uv run python -m review_app.manage add-reviewers maria.lopez jamal.k
    uv run python -m review_app.manage status               # progress counts
    uv run python -m review_app.manage export               # CSVs in output/review_exports/
"""

import argparse
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from review_app.app import normalize_id
from review_app.data import load_subset_items
from review_app.store import FirestoreStore, image_documents

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = REPO_ROOT / "output" / "subsets"
EXPORT_DIR = REPO_ROOT / "output" / "review_exports"
BATCH_LIMIT = 400


def _write_all(store: FirestoreStore, collection: str, documents: dict[str, dict]) -> None:
    batch, pending = store.client.batch(), 0
    for doc_id, data in documents.items():
        batch.set(store.client.collection(collection).document(doc_id), data)
        pending += 1
        if pending == BATCH_LIMIT:
            batch.commit()
            batch, pending = store.client.batch(), 0
    if pending:
        batch.commit()


def seed(store: FirestoreStore) -> None:
    if list(store.client.collection("items").limit(1).stream()):
        raise SystemExit("Firestore already has items; refusing to overwrite review progress.")
    items = load_subset_items(SUBSET_DIR)
    _write_all(store, "items", items)
    _write_all(store, "images", image_documents(items))
    logger.info("Loaded %s questions on %s images", len(items), len(image_documents(items)))


def add_reviewers(store: FirestoreStore, reviewer_ids: list[str]) -> None:
    for reviewer in (normalize_id(name) for name in reviewer_ids):
        reference = store.client.collection("annotators").document(reviewer)
        if reference.get().exists:
            logger.info("%s already exists", reviewer)
            continue
        reference.set({"reviewed": 0})
        logger.info("Added %s", reviewer)


def status(store: FirestoreStore) -> None:
    items = [snapshot.to_dict() for snapshot in store.client.collection("items").stream()]
    print("Items by status:", dict(Counter(item["status"] for item in items)))
    print("Edited versions created:", sum(item.get("source") == "modified" for item in items))
    for snapshot in store.client.collection("annotators").stream():
        print(f"  {snapshot.id}: {snapshot.to_dict().get('reviewed', 0)} reviewed")


def export(store: FirestoreStore) -> None:
    folder = EXPORT_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    for collection in ("items", "reviews"):
        rows = [
            {"id": snapshot.id, **snapshot.to_dict()}
            for snapshot in store.client.collection(collection).stream()
        ]
        frame = pd.DataFrame(rows)
        if collection == "items" and not frame.empty:
            frame = frame.drop(columns=["models"])
        frame.to_csv(folder / f"{collection}.csv", index=False)
        logger.info("Wrote %s %s", len(frame), collection)
    logger.info("Exported to %s", folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed")
    add = commands.add_parser("add-reviewers")
    add.add_argument("reviewer_ids", nargs="+")
    commands.add_parser("status")
    commands.add_parser("export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env.deploy")
    store = FirestoreStore(
        project=os.environ["GCP_PROJECT"],
        database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
    )
    if args.command == "seed":
        seed(store)
    elif args.command == "add-reviewers":
        add_reviewers(store, args.reviewer_ids)
    elif args.command == "status":
        status(store)
    else:
        export(store)


if __name__ == "__main__":
    main()
