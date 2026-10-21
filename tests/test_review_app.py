import pytest
from fastapi.testclient import TestClient

from review_app.app import create_app
from review_app.service import ReviewService
from review_app.store import MemoryStore


def _item(order: int, models_correct: int, question: str) -> dict:
    return {
        "image": f"img{order}.png",
        "image_url": f"https://example.com/img{order}.png",
        "problem": 1065658,
        "question": question,
        "answer": f"Teacher says {question}",
        "category": "Counting content",
        "models_correct": models_correct,
        "models": [{"name": "GPT-5", "answer": "No response provided.", "rating": 1}],
        "status": "pending",
        "source": "subset",
        "order": order,
    }


@pytest.fixture
def client():
    store = MemoryStore()
    store.seed(
        {"q1": _item(1, 0, "Hardest question?"), "q2": _item(2, 3, "Easier question?")},
        annotators=["maria.lopez"],
    )
    app = create_app(ReviewService(store), session_secret="test-secret", https_only=False)
    return TestClient(app)


def _sign_in(client: TestClient, reviewer_id: str = "maria.lopez"):
    return client.post("/login", data={"reviewer_id": reviewer_id}, follow_redirects=False)


def test_review_requires_sign_in(client):
    response = client.get("/review", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_unknown_id_cannot_sign_in(client):
    response = _sign_in(client, "someone.else")

    assert response.status_code == 401
    assert "isn’t on the reviewer list" in response.text


def test_ids_are_matched_ignoring_case_and_spaces(client):
    assert _sign_in(client, "  Maria.Lopez ").status_code == 303


def test_review_page_shows_image_question_answer_and_model_answers(client):
    _sign_in(client)

    page = client.get("/review").text

    assert "Hardest question?" in page
    assert "Teacher says Hardest question?" in page
    assert "https://example.com/img1.png" in page
    assert "No response provided." in page
    assert "<dialog" in page and "open" in page.split("<dialog", 1)[1].split(">", 1)[0]


def _guide_opens(page: str) -> bool:
    return "open" in page.split("<dialog", 1)[1].split(">", 1)[0]


def test_guidelines_open_on_every_sign_in_but_not_on_later_pages(client):
    _sign_in(client)
    assert _guide_opens(client.get("/review").text)
    assert not _guide_opens(client.get("/review").text)

    client.post("/logout")
    _sign_in(client)

    assert _guide_opens(client.get("/review").text)


def test_model_answers_come_after_the_question_and_decision_buttons(client):
    _sign_in(client)

    page = client.get("/review").text

    approve_button = page.index('class="btn approve">Approve<')
    assert page.index("Teacher says Hardest question?") < approve_button
    assert approve_button < page.index("No response provided.")


def test_approve_moves_straight_to_the_next_question(client):
    _sign_in(client)
    client.get("/review")

    response = client.post("/review/q1", data={"decision": "approve"})

    assert "Easier question?" in response.text
    assert "1 reviewed" in response.text


def test_reject_without_reason_shows_the_error_on_the_same_question(client):
    _sign_in(client)
    client.get("/review")

    response = client.post("/review/q1", data={"decision": "reject", "reason": ""})

    assert "Add a one-line reason" in response.text
    assert "Hardest question?" in response.text


def test_end_of_queue_says_so(client):
    _sign_in(client)
    client.get("/review")
    client.post("/review/q1", data={"decision": "approve"})

    response = client.post("/review/q2", data={"decision": "approve"})

    assert "end of the queue" in response.text


def test_decision_without_holding_the_image_is_refused_and_says_why(client):
    _sign_in(client)

    response = client.post("/review/q1", data={"decision": "approve"})

    assert "someone else picked it up" in response.text
    assert "0 reviewed" in response.text
