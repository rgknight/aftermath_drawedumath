"""Score model answers with the paper's three-judge ensemble via each provider's Batch API.

Mirrors allenai/DrawEduMath scripts/pipeline/judges/ (judge_claude.py, judge_gemini.py,
judge_gpt4o.py) and evaluation/run_evaluation.py: same prompt, same per-judge request
settings, same response parsing, and the same ensemble (mode of valid ratings; ties go to
the lowest rating, which is what scipy.stats.mode did for the published numbers).

Usage (from the repo root), after batch_generate.py has written output/<run>/predictions.csv:
    uv run python code/judge.py build --run luna6_none
    uv run python code/judge.py submit --run luna6_none
    uv run python code/judge.py status --run luna6_none
    uv run python code/judge.py collect --run luna6_none
    uv run python code/judge.py merge --run luna6_none
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

JUDGES = {
    # Upstream used the alias "claude-sonnet-4-5"; this is the snapshot it resolves to.
    "claude": {"model": "claude-sonnet-4-5-20250929", "column": "Claude"},
    "gemini": {"model": "gemini-2.5-pro", "column": "Gemini"},
    # Upstream used the alias "gpt-4o"; pin the snapshot it currently points to.
    "openai": {"model": "gpt-4o-2024-08-06", "column": "Openai"},
}

# Verbatim from allenai/DrawEduMath scripts/pipeline/prompts.py.
JUDGE_PROMPT_TEMPLATE = """
Given the following inputs:
Question: {question}
Answer 1 (Ground Truth): {teacher_a}
Answer 2 (Model Output): {model_a}

Your task is to rate the quality of Answer 2, using Answer 1 as the ground truth for a perfect response. Use the following 4-point scale:


**4: Semantically Identical**
- Answer 2 conveys the exact same meaning as Answer 1.
- It is fully correct and complete. Wording differences are acceptable.

**3: Different but valid**
- Answer 2 is factually correct and answers the question, but omits some details, nuance, or key points found in Answer 1, or adds some additional, plausible insight.
- The core idea is right, but the substance is partial.

**2: Factually incorrect in an important way**
- Answer 2 attempts to answer the question, but is factually incorrect in a significant way.
- Or, it mixes correct information with fabricated/hallucinated details not supported by the question or Answer 1.

**1: Irrelevant or Wrong**
- Answer 2 completely fails to answer the question, is on a different topic, or is nonsensical.


Provide both the Likert rating followed by a brief explanation for your choice. Format the output as a valid parsable JSON like: {{"rating": 1-4, "reason": "Your brief justification here."}}"""

_JUDGEMENT_PATTERN = re.compile(r'\{[^{}]*"rating"[^{}]*"reason"[^{}]*\}')
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def build_judge_prompt(
    question: str, reference_answer: str, model_answer: str, escape_backslashes: bool
) -> str:
    if escape_backslashes:
        reference_answer = reference_answer.replace("\\", "\\\\")
        model_answer = model_answer.replace("\\", "\\\\")
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question, teacher_a=reference_answer, model_a=model_answer
    )


def parse_judge_text(text: str) -> tuple[int, str]:
    """Extract (rating, reason) the way upstream does; rating is -1 if unusable."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    match = _JUDGEMENT_PATTERN.search(text)
    if match:
        text = match.group(0)
    text = _INVALID_JSON_ESCAPE.sub(r"\\\\", text)
    try:
        judgement = json.loads(text)
        rating = int(float(judgement.get("rating", -1)))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as error:
        return -1, f"Parse error: {error} (text={text or 'EMPTY'})"
    if rating not in (1, 2, 3, 4):
        return -1, f"Rating out of range: {rating}"
    return rating, str(judgement.get("reason", ""))


def ensemble_rating(ratings: list[int]) -> int:
    """Most common valid rating; ties go to the lowest. -1 if no judge produced a rating."""
    valid = [rating for rating in ratings if rating in (1, 2, 3, 4)]
    if not valid:
        return -1
    counts = Counter(valid)
    top_count = max(counts.values())
    return min(rating for rating, count in counts.items() if count == top_count)


def score_from_rating(rating: int) -> int:
    """Binarize as the paper does: 3-4 correct, 1-2 incorrect, -1 missing."""
    if rating in (3, 4):
        return 1
    if rating in (1, 2):
        return 0
    return -1


def build_openai_line(custom_id: str, prompt: str, model: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
    }


def build_claude_request(custom_id: str, prompt: str, model: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 10000,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def build_gemini_line(custom_id: str, prompt: str) -> dict:
    return {"key": custom_id, "request": {"contents": [{"parts": [{"text": prompt}]}]}}


def parse_openai_line(line: dict) -> tuple[str, int, str]:
    response = line.get("response") or {}
    body = response.get("body") or {}
    if line.get("error") or response.get("status_code") != 200:
        error = line.get("error") or body.get("error") or {}
        return line["custom_id"], -1, f"API error: {error.get('message', 'unknown')}"
    choices = body.get("choices") or []
    if not choices:
        return line["custom_id"], -1, "No choices in response"
    text = choices[0].get("message", {}).get("content") or ""
    return (line["custom_id"], *parse_judge_text(text))


def parse_gemini_line(line: dict) -> tuple[str, int, str]:
    key = line.get("key", "")
    if "error" in line:
        return key, -1, f"API error: {line['error'].get('message', 'unknown')}"
    candidates = (line.get("response") or {}).get("candidates") or []
    if not candidates:
        return key, -1, "No candidates in response"
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts or "text" not in parts[0]:
        return key, -1, "No text in parts"
    return (key, *parse_judge_text(parts[0]["text"]))


def parse_claude_result(result: dict) -> tuple[str, int, str]:
    custom_id = result["custom_id"]
    outcome = result["result"]
    if outcome["type"] == "errored":
        message = ((outcome.get("error") or {}).get("error") or {}).get("message", "unknown")
        return custom_id, -1, f"API error: {message}"
    if outcome["type"] != "succeeded":
        return custom_id, -1, f"Result type: {outcome['type']}"
    content = outcome["message"].get("content") or []
    if not content or "text" not in content[0]:
        return custom_id, -1, "Empty response"
    return (custom_id, *parse_judge_text(content[0]["text"]))


def _judge_dir(run_dir: Path, judge: str) -> Path:
    return run_dir / "judges" / judge


def build_run(run_dir: Path) -> None:
    predictions = pd.read_csv(run_dir / "predictions.csv", keep_default_na=False)
    fields = predictions[["custom_id", "Question", "Reference Answer", "Model Answer"]]
    fields = fields.map(lambda value: str(value).strip())
    judgeable = fields[(fields != "").all(axis=1)]
    logger.info("%s of %s rows have question, reference and answer", len(judgeable), len(fields))

    for judge, config in JUDGES.items():
        judge_dir = _judge_dir(run_dir, judge)
        judge_dir.mkdir(parents=True, exist_ok=True)
        with open(judge_dir / "requests.jsonl", "w") as outfile:
            for row in judgeable.itertuples(index=False):
                prompt = build_judge_prompt(
                    row.Question,
                    row[2],  # Reference Answer
                    row[3],  # Model Answer
                    escape_backslashes=(judge == "claude"),
                )
                if judge == "claude":
                    line = build_claude_request(row.custom_id, prompt, config["model"])
                elif judge == "gemini":
                    line = build_gemini_line(row.custom_id, prompt)
                else:
                    line = build_openai_line(row.custom_id, prompt, config["model"])
                outfile.write(json.dumps(line) + "\n")
        logger.info("Wrote %s requests for %s", len(judgeable), judge)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as infile:
        return [json.loads(line) for line in infile if line.strip()]


def _clients(judges: list[str]) -> dict:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    clients = {}
    if "claude" in judges:
        import anthropic

        clients["claude"] = anthropic.Anthropic()
    if "gemini" in judges:
        from google import genai

        clients["gemini"] = genai.Client()
    if "openai" in judges:
        from openai import OpenAI

        clients["openai"] = OpenAI()
    return clients


def submit_judge(run_dir: Path, judge: str, client) -> None:
    judge_dir = _judge_dir(run_dir, judge)
    batch_file = judge_dir / "batch.json"
    if batch_file.exists():
        logger.info("%s already submitted (%s); skipping", judge, batch_file)
        return
    requests_path = judge_dir / "requests.jsonl"
    model = JUDGES[judge]["model"]
    if judge == "claude":
        batch = client.messages.batches.create(requests=_read_jsonl(requests_path))
        record = {"batch_id": batch.id}
    elif judge == "gemini":
        from google.genai import types

        uploaded = client.files.upload(
            file=str(requests_path),
            config=types.UploadFileConfig(display_name=f"{run_dir.name}-judge", mime_type="jsonl"),
        )
        batch = client.batches.create(
            model=model, src=uploaded.name, config={"display_name": f"{run_dir.name}-judge"}
        )
        record = {"batch_id": batch.name, "input_file": uploaded.name}
    else:
        with open(requests_path, "rb") as infile:
            uploaded = client.files.create(file=infile, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"run": run_dir.name, "judge": judge},
        )
        record = {"batch_id": batch.id, "input_file_id": uploaded.id}
    record["model"] = model
    batch_file.write_text(json.dumps(record))
    logger.info("Submitted %s judge batch %s", judge, record["batch_id"])


def get_status(run_dir: Path, judge: str, client) -> tuple[str, bool, str]:
    """Return (state, finished, detail) for a judge's batch."""
    batch_id = json.loads((_judge_dir(run_dir, judge) / "batch.json").read_text())["batch_id"]
    if judge == "claude":
        batch = client.messages.batches.retrieve(batch_id)
        return (
            batch.processing_status,
            batch.processing_status == "ended",
            str(batch.request_counts),
        )
    if judge == "gemini":
        batch = client.batches.get(name=batch_id)
        state = batch.state.name
        finished = state in {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        return state, finished, ""
    batch = client.batches.retrieve(batch_id)
    finished = batch.status in {"completed", "failed", "expired", "cancelled"}
    return batch.status, finished, str(batch.request_counts)


def collect_judge(run_dir: Path, judge: str, client) -> None:
    judge_dir = _judge_dir(run_dir, judge)
    batch_id = json.loads((judge_dir / "batch.json").read_text())["batch_id"]
    if judge == "claude":
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            raise RuntimeError(f"Claude batch {batch_id} is {batch.processing_status}")
        parsed = [
            parse_claude_result(entry.to_dict())
            for entry in client.messages.batches.results(batch_id)
        ]
    elif judge == "gemini":
        batch = client.batches.get(name=batch_id)
        if batch.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch {batch_id} is {batch.state.name}")
        content = client.files.download(file=batch.dest.file_name).decode("utf-8")
        parsed = [
            parse_gemini_line(json.loads(line)) for line in content.splitlines() if line.strip()
        ]
    else:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch {batch_id} is {batch.status}")
        lines = []
        for file_id in [batch.output_file_id, batch.error_file_id]:
            if file_id:
                text = client.files.content(file_id).text
                lines.extend(json.loads(line) for line in text.splitlines() if line.strip())
        parsed = [parse_openai_line(line) for line in lines]

    ratings = pd.DataFrame(parsed, columns=["custom_id", "Judge_Rating", "Judge_Reason"])
    ratings.to_csv(judge_dir / "ratings.csv", index=False)
    logger.info(
        "%s: %s results, ratings %s",
        judge,
        len(ratings),
        ratings["Judge_Rating"].value_counts().sort_index().to_dict(),
    )


def merge_run(run_dir: Path) -> None:
    judged = pd.read_csv(run_dir / "predictions.csv")
    rating_columns = []
    for judge, config in JUDGES.items():
        ratings = pd.read_csv(_judge_dir(run_dir, judge) / "ratings.csv")
        column = config["column"]
        ratings = ratings.rename(
            columns={
                "Judge_Rating": f"{column}_Judge_Rating",
                "Judge_Reason": f"{column}_Judge_Reason",
            }
        )
        judged = judged.merge(ratings, on="custom_id", how="left", validate="one_to_one")
        judged[f"{column}_Judge_Rating"] = judged[f"{column}_Judge_Rating"].fillna(-1).astype(int)
        rating_columns.append(f"{column}_Judge_Rating")

    judged["Ensemble_Judge_Rating"] = judged[rating_columns].apply(
        lambda row: ensemble_rating(row.tolist()), axis=1
    )
    judged["Score"] = judged["Ensemble_Judge_Rating"].map(score_from_rating)
    judged.to_csv(run_dir / "judged.csv", index=False)
    logger.info("Missing ratings per judge: %s", (judged[rating_columns] == -1).sum().to_dict())
    logger.info(
        "Ensemble ratings: %s",
        judged["Ensemble_Judge_Rating"].value_counts().sort_index().to_dict(),
    )
    logger.info("Wrote %s", run_dir / "judged.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "submit", "status", "collect", "merge"])
    parser.add_argument("--run", required=True, help="Run name; reads output/<run>/predictions.csv")
    parser.add_argument("--judge", choices=[*JUDGES, "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ["httpx", "httpx2", "httpcore", "google_genai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run_dir = OUTPUT_DIR / args.run
    judges = list(JUDGES) if args.judge == "all" else [args.judge]

    if args.command == "build":
        build_run(run_dir)
        return
    if args.command == "merge":
        merge_run(run_dir)
        return

    clients = _clients(judges)
    for judge in judges:
        if args.command == "submit":
            submit_judge(run_dir, judge, clients[judge])
        elif args.command == "status":
            state, finished, detail = get_status(run_dir, judge, clients[judge])
            logger.info("%s: %s%s %s", judge, state, " (finished)" if finished else "", detail)
        elif args.command == "collect":
            collect_judge(run_dir, judge, clients[judge])


if __name__ == "__main__":
    main()
