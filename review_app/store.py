"""Document storage for the review app: Firestore in production, in memory for tests and local use.

Both stores expose the same small interface. Transactions follow Firestore's rules: every
read (get, query) happens before any write, and the function may be retried, so it must not
have side effects outside the transaction.
"""

import copy
import itertools
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")
Rows = list[tuple[str, dict[str, Any]]]


class Transaction(Protocol):
    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...
    def query(self, collection: str, where: dict[str, Any]) -> Rows: ...
    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None: ...
    def add(self, collection: str, data: dict[str, Any]) -> str: ...


class Store(Protocol):
    def run_transaction(self, function: Callable[[Transaction], T]) -> T: ...
    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...
    def query(
        self,
        collection: str,
        where: dict[str, Any],
        order_by: str | None = None,
        limit: int | None = None,
    ) -> Rows: ...
    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None: ...


def image_documents(items: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Queue state per image, derived from its pending items."""
    images: dict[str, dict[str, Any]] = {}
    for item in items.values():
        if item["status"] != "pending":
            continue
        image = images.setdefault(
            item["image"],
            {
                "problem": item["problem"],
                "pending": 0,
                "priority": item["models_correct"],
                "open": True,
                "lock_by": None,
                "lock_until": None,
                "skipped_by": [],
            },
        )
        image["pending"] += 1
        image["priority"] = min(image["priority"], item["models_correct"])
    return images


def _matches(document: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(document.get(field) == value for field, value in where.items())


class MemoryStore:
    """Thread-safe in-memory store for tests and local runs."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._ids = itertools.count(1)

    def seed(self, items: dict[str, dict[str, Any]], annotators: list[str]) -> None:
        with self._lock:
            self._collections["items"].update(copy.deepcopy(items))
            self._collections["images"].update(image_documents(items))
            for annotator in annotators:
                self._collections["annotators"][annotator] = {"reviewed": 0}

    def new_id(self) -> str:
        return f"auto{next(self._ids):06d}"

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            document = self._collections[collection].get(doc_id)
            return copy.deepcopy(document) if document is not None else None

    def query(
        self,
        collection: str,
        where: dict[str, Any],
        order_by: str | None = None,
        limit: int | None = None,
    ) -> Rows:
        with self._lock:
            rows = [
                (doc_id, copy.deepcopy(document))
                for doc_id, document in self._collections[collection].items()
                if _matches(document, where)
            ]
        if order_by:
            rows.sort(key=lambda row: row[1].get(order_by))
        return rows[:limit] if limit else rows

    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            self._collections[collection][doc_id].update(copy.deepcopy(fields))

    def run_transaction(self, function: Callable[[Transaction], T]) -> T:
        with self._lock:
            transaction = _MemoryTransaction(self)
            result = function(transaction)
            transaction.commit()
            return result


class _MemoryTransaction:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._writes: list[Callable[[], None]] = []

    def _check_read(self) -> None:
        if self._writes:
            raise RuntimeError("Transactions must do all reads before any writes")

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        self._check_read()
        return self._store.get(collection, doc_id)

    def query(self, collection: str, where: dict[str, Any]) -> Rows:
        self._check_read()
        return self._store.query(collection, where)

    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None:
        self._writes.append(lambda: self._store.update(collection, doc_id, fields))

    def add(self, collection: str, data: dict[str, Any]) -> str:
        doc_id = self._store.new_id()
        collections = self._store._collections
        self._writes.append(
            lambda: collections[collection].__setitem__(doc_id, copy.deepcopy(data))
        )
        return doc_id

    def commit(self) -> None:
        for write in self._writes:
            write()


class FirestoreStore:
    """Firestore (native mode) store. Needs a composite index on images(open, priority)."""

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        self._firestore = firestore
        self.client = firestore.Client(project=project, database=database)

    def _query(self, collection: str, where: dict[str, Any]):
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = self.client.collection(collection)
        for field, value in where.items():
            query = query.where(filter=FieldFilter(field, "==", value))
        return query

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        snapshot = self.client.collection(collection).document(doc_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def query(
        self,
        collection: str,
        where: dict[str, Any],
        order_by: str | None = None,
        limit: int | None = None,
    ) -> Rows:
        query = self._query(collection, where)
        if order_by:
            query = query.order_by(order_by)
        if limit:
            query = query.limit(limit)
        return [(snapshot.id, snapshot.to_dict()) for snapshot in query.stream()]

    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None:
        self.client.collection(collection).document(doc_id).update(fields)

    def run_transaction(self, function: Callable[[Transaction], T]) -> T:
        store = self

        @self._firestore.transactional
        def run(transaction):
            return function(_FirestoreTransaction(store, transaction))

        return run(self.client.transaction())


class _FirestoreTransaction:
    def __init__(self, store: FirestoreStore, transaction) -> None:
        self._store = store
        self._transaction = transaction

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        reference = self._store.client.collection(collection).document(doc_id)
        snapshot = reference.get(transaction=self._transaction)
        return snapshot.to_dict() if snapshot.exists else None

    def query(self, collection: str, where: dict[str, Any]) -> Rows:
        query = self._store._query(collection, where)
        return [(s.id, s.to_dict()) for s in query.stream(transaction=self._transaction)]

    def update(self, collection: str, doc_id: str, fields: dict[str, Any]) -> None:
        reference = self._store.client.collection(collection).document(doc_id)
        self._transaction.update(reference, fields)

    def add(self, collection: str, data: dict[str, Any]) -> str:
        reference = self._store.client.collection(collection).document()
        self._transaction.create(reference, data)
        return reference.id
