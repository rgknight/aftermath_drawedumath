"""Teacher review app: approve, reject or modify hard DrawEduMath questions, one at a time.

Run locally against the subset in output/subsets (no GCP needed):
    REVIEW_STORE=memory DEV_ANNOTATORS=maria.lopez,jamal.k \\
        uv run uvicorn review_app.app:create_app_from_env --factory --port 8766

On Cloud Run it uses Firestore; see docs/review_app.md.
"""

import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from review_app.service import LEASE_SECONDS, AlreadyReviewed, LockLost, ReviewService

logger = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
RATING_LABELS = {
    4: "Same meaning",
    3: "Correct, partial",
    2: "Wrong",
    1: "Irrelevant or wrong",
    -1: "No rating",
}
SESSION_SECONDS = 12 * 60 * 60


def normalize_id(reviewer_id: str) -> str:
    return reviewer_id.strip().lower()


def create_app(service: ReviewService, session_secret: str, https_only: bool = True) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        max_age=SESSION_SECONDS,
        same_site="lax",
        https_only=https_only,
    )

    def signed_in(request: Request) -> str | None:
        reviewer = request.session.get("reviewer")
        return reviewer if reviewer and service.is_annotator(reviewer) else None

    def to_review() -> RedirectResponse:
        return RedirectResponse("/review", status_code=303)

    @app.get("/")
    def index() -> Response:
        return to_review()

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        return TEMPLATES.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    def login(request: Request, reviewer_id: str = Form("")) -> Response:
        reviewer = normalize_id(reviewer_id)
        if not reviewer or not service.is_annotator(reviewer):
            error = "That ID isn’t on the reviewer list. Check it, or ask the project lead."
            return TEMPLATES.TemplateResponse(
                request, "login.html", {"error": error}, status_code=401
            )
        request.session.clear()
        request.session["reviewer"] = reviewer
        request.session["show_guide"] = True
        logger.info("Reviewer %s signed in", reviewer)
        return to_review()

    @app.post("/logout")
    def logout(request: Request) -> Response:
        reviewer = signed_in(request)
        if reviewer:
            service.release(reviewer)
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/review", response_class=HTMLResponse)
    def review(request: Request) -> Response:
        reviewer = signed_in(request)
        if not reviewer:
            return RedirectResponse("/login", status_code=303)
        context = {
            "reviewer": reviewer,
            "reviewed": service.annotator(reviewer).get("reviewed", 0),
            "flash": request.session.pop("flash", None),
            "form": request.session.pop("form", None),
            "show_guide": request.session.pop("show_guide", False),
            "lease_minutes": LEASE_SECONDS // 60,
            "rating_labels": RATING_LABELS,
        }
        next_item = service.next_item(reviewer)
        if next_item is None:
            return TEMPLATES.TemplateResponse(request, "done.html", context)
        item_id, item = next_item
        return TEMPLATES.TemplateResponse(
            request, "review.html", {**context, "item_id": item_id, "item": item}
        )

    @app.post("/review/{item_id}")
    def decide(
        request: Request,
        item_id: str,
        decision: str = Form(""),
        reason: str = Form(""),
        question: str = Form(""),
        answer: str = Form(""),
    ) -> Response:
        reviewer = signed_in(request)
        if not reviewer:
            return RedirectResponse("/login", status_code=303)
        try:
            service.decide(reviewer, item_id, decision, reason, question, answer)
        except ValueError as error:
            request.session["flash"] = {"kind": "error", "text": str(error)}
            request.session["form"] = {
                "item": item_id,
                "decision": decision,
                "reason": reason,
                "question": question,
                "answer": answer,
            }
        except LockLost:
            request.session["flash"] = {
                "kind": "notice",
                "text": "Your hold on that image ran out and someone else picked it up. "
                "Here’s your next question.",
            }
        except AlreadyReviewed:
            request.session["flash"] = {
                "kind": "notice",
                "text": "That question was already recorded.",
            }
        except KeyError:
            request.session["flash"] = {"kind": "notice", "text": "That question no longer exists."}
        return to_review()

    @app.post("/skip")
    def skip(request: Request) -> Response:
        reviewer = signed_in(request)
        if not reviewer:
            return RedirectResponse("/login", status_code=303)
        service.skip(reviewer)
        request.session["flash"] = {
            "kind": "notice",
            "text": "Image skipped. It’s back in the queue for someone else.",
        }
        return to_review()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def create_app_from_env() -> FastAPI:
    """Build the app from environment variables (used by uvicorn --factory)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backend = os.environ.get("REVIEW_STORE", "firestore")
    if backend == "memory":
        from review_app.data import load_subset_items
        from review_app.store import MemoryStore

        repo_root = Path(__file__).resolve().parent.parent
        subset_dir = Path(os.environ.get("SUBSET_DIR", repo_root / "output" / "subsets"))
        store = MemoryStore()
        annotators = [
            normalize_id(name)
            for name in os.environ.get("DEV_ANNOTATORS", "").split(",")
            if name.strip()
        ]
        store.seed(load_subset_items(subset_dir), annotators=annotators)
        logger.info("In-memory store with reviewers: %s", ", ".join(annotators) or "(none)")
        session_secret = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(32)
        return create_app(ReviewService(store), session_secret, https_only=False)

    from review_app.store import FirestoreStore

    store = FirestoreStore(
        project=os.environ.get("GCP_PROJECT"),
        database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
    )
    return create_app(ReviewService(store), os.environ["SESSION_SECRET"], https_only=True)
