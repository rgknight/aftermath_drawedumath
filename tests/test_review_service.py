import pytest

from review_app.service import LEASE_SECONDS, LockLost, ReviewService
from review_app.store import MemoryStore


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _item(image: str, order: int, models_correct: int, question: str) -> dict:
    return {
        "image": image,
        "image_url": f"https://example.com/{image}",
        "problem": 1,
        "question": question,
        "answer": f"answer to {question}",
        "category": "Counting content",
        "models_correct": models_correct,
        "models": [{"name": "Model A", "answer": "a", "rating": 2}],
        "status": "pending",
        "source": "subset",
        "order": order,
    }


@pytest.fixture
def setup():
    store = MemoryStore()
    items = {
        "q1": _item("easy.png", 1, 4, "easy question"),
        "q2": _item("hard.png", 2, 3, "hard image, easier question"),
        "q3": _item("hard.png", 3, 0, "hardest question"),
        "q4": _item("mid.png", 4, 2, "mid question"),
    }
    store.seed(items, annotators=["ana", "ben"])
    clock = Clock()
    return ReviewService(store, clock=clock), store, clock


def test_serves_hardest_image_first_and_its_hardest_question_first(setup):
    service, _, _ = setup

    item_id, item = service.next_item("ana")

    assert item_id == "q3"
    assert item["image"] == "hard.png"


def test_two_annotators_never_share_an_image(setup):
    service, _, _ = setup

    _, first = service.next_item("ana")
    _, second = service.next_item("ben")

    assert first["image"] == "hard.png"
    assert second["image"] == "mid.png"


def test_annotator_keeps_their_image_until_its_questions_are_done(setup):
    service, _, _ = setup
    service.next_item("ana")

    service.decide("ana", "q3", "approve")
    item_id, _ = service.next_item("ana")

    assert item_id == "q2"


def test_lock_expires_after_an_hour_and_the_late_decision_is_refused(setup):
    service, _, clock = setup
    service.next_item("ana")

    clock.now += LEASE_SECONDS + 1
    item_id, _ = service.next_item("ben")

    assert item_id == "q3"
    with pytest.raises(LockLost):
        service.decide("ana", "q3", "approve")


def test_activity_renews_the_lock(setup):
    service, _, clock = setup
    service.next_item("ana")
    clock.now += LEASE_SECONDS - 60
    service.decide("ana", "q3", "approve")

    clock.now += LEASE_SECONDS - 60
    _, other = service.next_item("ben")

    assert other["image"] == "mid.png"


def test_reject_requires_a_reason(setup):
    service, _, _ = setup
    service.next_item("ana")

    with pytest.raises(ValueError):
        service.decide("ana", "q3", "reject", reason="  ")

    service.decide("ana", "q3", "reject", reason="Reference answer is wrong")
    assert service.store.get("items", "q3")["status"] == "rejected"


def test_modify_creates_a_pending_version_without_model_answers(setup):
    service, store, _ = setup
    service.next_item("ana")

    new_id = service.decide(
        "ana", "q3", "modify", question="Clearer question?", answer="Clearer answer"
    )

    assert store.get("items", "q3")["status"] == "modified"
    new = store.get("items", new_id)
    assert new["status"] == "pending"
    assert new["models"] == []
    assert new["parent"] == "q3"
    assert new["created_by"] == "ana"
    assert (new["image"], new["question"], new["answer"]) == (
        "hard.png",
        "Clearer question?",
        "Clearer answer",
    )


def test_modify_without_changes_is_refused(setup):
    service, _, _ = setup
    _, item = service.next_item("ana")

    with pytest.raises(ValueError):
        service.decide("ana", "q3", "modify", question=item["question"], answer=item["answer"])


def test_the_author_of_a_modification_is_not_served_it_but_others_are(setup):
    service, _, _ = setup
    service.next_item("ana")
    new_id = service.decide("ana", "q3", "modify", question="New q?", answer="New a")

    served_to_ana = []
    while (next_item := service.next_item("ana")) is not None:
        item_id, _ = next_item
        served_to_ana.append(item_id)
        service.decide("ana", item_id, "approve")

    assert new_id not in served_to_ana
    assert service.next_item("ben")[0] == new_id


def test_skip_returns_the_image_to_the_queue_for_someone_else(setup):
    service, _, _ = setup
    service.next_item("ana")

    service.skip("ana")

    assert service.next_item("ana")[1]["image"] == "mid.png"
    assert service.next_item("ben")[1]["image"] == "hard.png"


def test_returns_none_when_the_queue_is_empty(setup):
    service, _, _ = setup
    while (next_item := service.next_item("ana")) is not None:
        service.decide("ana", next_item[0], "approve")

    assert service.next_item("ana") is None
    assert service.next_item("ben") is None


def test_counts_reviews_per_annotator(setup):
    service, store, _ = setup
    service.next_item("ana")
    service.decide("ana", "q3", "approve")
    service.decide("ana", "q2", "reject", reason="Ambiguous")

    assert store.get("annotators", "ana")["reviewed"] == 2


def test_memory_store_refuses_reads_after_writes_like_firestore():
    store = MemoryStore()
    store.seed({}, annotators=["ana"])

    def read_after_write(txn):
        txn.update("annotators", "ana", {"reviewed": 1})
        txn.get("annotators", "ana")

    with pytest.raises(RuntimeError):
        store.run_transaction(read_after_write)
