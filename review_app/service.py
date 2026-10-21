"""Review queue: who reviews which image next, and what each decision does.

- Annotators claim a whole image and review its pending questions hardest first (fewest of the
  11 published models correct). Images are claimed hardest first too.
- A claim is a lease that runs LEASE_SECONDS from the annotator's last action. After it lapses,
  anyone else can claim the image and the original annotator's late decisions are refused.
- Modify closes the item and adds a new pending version on the same image, with no model
  answers. Its author is never served it; anyone else can review it.
- Skip hands the image back to the queue; the annotator who skipped it isn't offered it again.
"""

import time
from collections.abc import Callable
from typing import Any

from review_app.store import Store, Transaction

LEASE_SECONDS = 60 * 60
MAX_REASON_LENGTH = 300
MAX_TEXT_LENGTH = 2000
DECISION_STATUS = {"approve": "approved", "reject": "rejected", "modify": "modified"}
NO_PENDING_PRIORITY = 99
CANDIDATE_PAGE_SIZES = (25, 200, None)

Item = tuple[str, dict[str, Any]]


class LockLost(Exception):
    """The annotator no longer holds the image this item belongs to."""


class AlreadyReviewed(Exception):
    """The item was decided already (for example, a double-submitted form)."""


def _item_order(row: Item) -> tuple[int, int]:
    return row[1]["models_correct"], row[1]["order"]


def _servable(items: list[Item], annotator: str) -> list[Item]:
    return sorted((row for row in items if row[1].get("created_by") != annotator), key=_item_order)


def _is_locked_by_other(image: dict[str, Any], annotator: str, now: float) -> bool:
    holder = image.get("lock_by")
    return bool(holder) and holder != annotator and (image.get("lock_until") or 0) > now


class ReviewService:
    def __init__(self, store: Store, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.clock = clock

    def is_annotator(self, annotator: str) -> bool:
        return self.store.get("annotators", annotator) is not None

    def annotator(self, annotator: str) -> dict[str, Any]:
        return self.store.get("annotators", annotator) or {}

    def next_item(self, annotator: str) -> Item | None:
        """The annotator's next question, claiming a new image when theirs is finished."""
        now = self.clock()
        for image_id, _ in self.store.query("images", {"lock_by": annotator}):
            item = self.store.run_transaction(
                lambda txn, image_id=image_id: self._continue(txn, image_id, annotator, now)
            )
            if item:
                return item

        for page_size in CANDIDATE_PAGE_SIZES:
            candidates = self.store.query(
                "images", {"open": True}, order_by="priority", limit=page_size
            )
            candidates.sort(key=lambda row: (row[1]["priority"], row[1]["problem"], row[0]))
            for image_id, image in candidates:
                if annotator in image.get("skipped_by", []):
                    continue
                if _is_locked_by_other(image, annotator, now):
                    continue
                item = self.store.run_transaction(
                    lambda txn, image_id=image_id: self._claim(txn, image_id, annotator, now)
                )
                if item:
                    return item
            if page_size is None or len(candidates) < page_size:
                return None
        return None

    def _continue(self, txn: Transaction, image_id: str, annotator: str, now: float) -> Item | None:
        image = txn.get("images", image_id)
        if not image or image.get("lock_by") != annotator:
            return None
        servable = _servable(
            txn.query("items", {"image": image_id, "status": "pending"}), annotator
        )
        if not servable:
            txn.update("images", image_id, {"lock_by": None, "lock_until": None})
            return None
        txn.update("images", image_id, {"lock_until": now + LEASE_SECONDS})
        return servable[0]

    def _claim(self, txn: Transaction, image_id: str, annotator: str, now: float) -> Item | None:
        image = txn.get("images", image_id)
        if not image or not image.get("open") or annotator in image.get("skipped_by", []):
            return None
        if _is_locked_by_other(image, annotator, now):
            return None
        servable = _servable(
            txn.query("items", {"image": image_id, "status": "pending"}), annotator
        )
        if not servable:
            return None
        txn.update("images", image_id, {"lock_by": annotator, "lock_until": now + LEASE_SECONDS})
        return servable[0]

    def decide(
        self,
        annotator: str,
        item_id: str,
        decision: str,
        reason: str = "",
        question: str = "",
        answer: str = "",
    ) -> str | None:
        """Record approve / reject / modify. Returns the new item's id for modify."""
        if decision not in DECISION_STATUS:
            raise ValueError(f"Unknown decision: {decision}")
        reason, question, answer = reason.strip(), question.strip(), answer.strip()
        if decision == "reject" and not reason:
            raise ValueError("Add a one-line reason for rejecting this question.")
        if decision == "modify" and not (question and answer):
            raise ValueError("A modified question needs both a question and an answer.")
        if len(reason) > MAX_REASON_LENGTH:
            raise ValueError(f"Keep the reason under {MAX_REASON_LENGTH} characters.")
        if max(len(question), len(answer)) > MAX_TEXT_LENGTH:
            raise ValueError(f"Keep the question and answer under {MAX_TEXT_LENGTH} characters.")
        now = self.clock()

        def record(txn: Transaction) -> str | None:
            item = txn.get("items", item_id)
            if item is None:
                raise KeyError(item_id)
            if item["status"] != "pending":
                raise AlreadyReviewed(item_id)
            image_id = item["image"]
            image = txn.get("images", image_id) or {}
            if image.get("lock_by") != annotator:
                raise LockLost(image_id)
            if decision == "modify" and (question, answer) == (item["question"], item["answer"]):
                raise ValueError("Nothing was changed. Approve the question instead.")
            reviewer = txn.get("annotators", annotator) or {}
            pending = txn.query("items", {"image": image_id, "status": "pending"})

            remaining = [row for row in pending if row[0] != item_id]
            txn.update(
                "items",
                item_id,
                {"status": DECISION_STATUS[decision], "reviewed_by": annotator, "reviewed_at": now},
            )
            review = {
                "item": item_id,
                "image": image_id,
                "annotator": annotator,
                "decision": decision,
                "reason": reason,
                "at": now,
            }
            new_id = None
            if decision == "modify":
                new_item = {
                    **{key: item[key] for key in ("image", "image_url", "problem", "category")},
                    "question": question,
                    "answer": answer,
                    "models_correct": item["models_correct"],
                    "models": [],
                    "status": "pending",
                    "source": "modified",
                    "parent": item_id,
                    "created_by": annotator,
                    "created_at": now,
                    "order": item["order"],
                }
                new_id = txn.add("items", new_item)
                remaining.append((new_id, new_item))
                review.update({"question": question, "answer": answer, "new_item": new_id})
            txn.add("reviews", review)
            txn.update(
                "images",
                image_id,
                {
                    "pending": len(remaining),
                    "open": bool(remaining),
                    "priority": min(
                        (row[1]["models_correct"] for row in remaining), default=NO_PENDING_PRIORITY
                    ),
                    "lock_until": now + LEASE_SECONDS,
                },
            )
            txn.update("annotators", annotator, {"reviewed": reviewer.get("reviewed", 0) + 1})
            return new_id

        return self.store.run_transaction(record)

    def skip(self, annotator: str) -> None:
        """Hand the annotator's current image back to the queue for someone else."""
        self._release(annotator, skipped=True)

    def release(self, annotator: str) -> None:
        """Free the annotator's image without skipping it (for example, on sign-out)."""
        self._release(annotator, skipped=False)

    def _release(self, annotator: str, skipped: bool) -> None:
        now = self.clock()
        for image_id, _ in self.store.query("images", {"lock_by": annotator}):

            def free(txn: Transaction, image_id: str = image_id) -> None:
                image = txn.get("images", image_id)
                if not image or image.get("lock_by") != annotator:
                    return
                fields: dict[str, Any] = {"lock_by": None, "lock_until": None}
                if skipped:
                    fields["skipped_by"] = [*image.get("skipped_by", []), annotator]
                txn.update("images", image_id, fields)
                if skipped:
                    txn.add(
                        "reviews",
                        {"image": image_id, "annotator": annotator, "decision": "skip", "at": now},
                    )

            self.store.run_transaction(free)
