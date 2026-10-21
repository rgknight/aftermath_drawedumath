"""Generate DrawEduMath answers for a new model with the OpenAI Batch API.

Mirrors allenai/DrawEduMath scripts/pipeline/generation/generate_openai.py (same system
prompt and user turn) but sends requests through the Responses API in one batch and
references images by their public S3 URL instead of base64.

Usage (from the repo root):
    uv run python code/batch_generate.py build --run luna6_none
    uv run python code/batch_generate.py submit --run luna6_none
    uv run python code/batch_generate.py status --run luna6_none
    uv run python code/batch_generate.py collect --run luna6_none

See docs/plans/2026_10_20_luna_error_eval.md for scope and differences from the paper.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# Every model in predictions.csv answered the same rows; take them from one model.
REFERENCE_MODEL = "GPT-5"
KEY_COLUMNS = ["Image Name", "QA Type", "Question", "Reference Answer"]
MANIFEST_COLUMNS = [
    "Problem ID",
    "QA Type",
    "Image Name",
    "Image URL",
    "Question",
    "Reference Answer",
]

# Verbatim from allenai/DrawEduMath scripts/pipeline/prompts.py.
GENERATE_ANSWER_PROMPT = """You are given an image of a student's handwritten work in response to a math problem.
The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
Your task is to answer a question based solely on the visual content of the student's handwritten work,
which is present on the right side of the image. Your answer should be clear and concise, and directly relate to the image presented
on the right side of the given image.
For example, given the question:
"What is the equation shown in the image?"
Generate your answer as: "3x + 2 = 8"
"""


def select_sample_rows(predictions: pd.DataFrame, sampled_images: list[str]) -> pd.DataFrame:
    """Return the QA rows for the sampled images, taken from one reference model."""
    sampled_stems = {Path(name).stem for name in sampled_images}
    rows = predictions[predictions["Model Name"] == REFERENCE_MODEL]
    rows = rows[rows["Image Name"].map(lambda name: Path(name).stem in sampled_stems)]
    return rows.drop(columns="Model Name").reset_index(drop=True)


def make_custom_id(row: pd.Series) -> str:
    """Stable id for a QA row; the four key columns are unique within the sample."""
    key = "\x1f".join(str(row[column]) for column in KEY_COLUMNS)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def build_request(
    row: pd.Series,
    custom_id: str,
    image_url: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict:
    """One Batch API line for the Responses endpoint."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": GENERATE_ANSWER_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Answer the following question: " + str(row["Question"]),
                        },
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
    }


def parse_result_line(line: dict) -> dict:
    """Flatten one Batch API output line into answer text, status, and token usage."""
    result = {
        "custom_id": line["custom_id"],
        "Model Answer": "",
        "Response Status": "error",
        "Incomplete Reason": "",
        "Error": "",
        "Input Tokens": None,
        "Output Tokens": None,
        "Reasoning Tokens": None,
    }
    response = line.get("response") or {}
    body = response.get("body") or {}
    if line.get("error") or response.get("status_code") != 200:
        error = line.get("error") or body.get("error") or {}
        result["Error"] = error.get("message", "unknown error")
        return result

    texts, refusals = [], []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part["text"])
            elif part.get("type") == "refusal":
                refusals.append(part["refusal"])

    usage = body.get("usage") or {}
    result["Model Answer"] = "".join(texts) if texts else "".join(refusals)
    result["Response Status"] = "refused" if refusals and not texts else body.get("status", "")
    result["Incomplete Reason"] = (body.get("incomplete_details") or {}).get("reason", "")
    result["Input Tokens"] = usage.get("input_tokens")
    result["Output Tokens"] = usage.get("output_tokens")
    result["Reasoning Tokens"] = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    return result


def build_run(run_dir: Path, model: str, reasoning_effort: str, max_output_tokens: int) -> None:
    predictions = pd.read_csv(DATA_DIR / "predictions.csv")
    with open(DATA_DIR / "sampled_images_per_problem.json") as infile:
        sampled_images = json.load(infile)["1"]
    image_urls = pd.read_csv(DATA_DIR / "DrawEduMath_QA.csv").set_index("Image Name")["Image URL"]

    rows = select_sample_rows(predictions, sampled_images)
    rows["Image URL"] = rows["Image Name"].map(image_urls)
    if rows["Image URL"].isna().any():
        raise ValueError(f"{rows['Image URL'].isna().sum()} rows have no image URL")
    rows["custom_id"] = rows.apply(make_custom_id, axis=1)
    if rows["custom_id"].duplicated().any():
        raise ValueError("custom_id collision; key columns are not unique")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "requests.jsonl", "w") as outfile:
        for _, row in rows.iterrows():
            request = build_request(
                row, row["custom_id"], row["Image URL"], model, reasoning_effort, max_output_tokens
            )
            outfile.write(json.dumps(request) + "\n")
    rows[["custom_id", *MANIFEST_COLUMNS]].to_csv(run_dir / "manifest.csv", index=False)
    config = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "requests": len(rows),
        "images": int(rows["Image Name"].nunique()),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    logger.info("Wrote %s requests for %s images to %s", len(rows), config["images"], run_dir)


def submit_run(run_dir: Path, client) -> None:
    batch_file = run_dir / "batch.json"
    if batch_file.exists():
        raise FileExistsError(f"{batch_file} exists; this run was already submitted")
    with open(run_dir / "requests.jsonl", "rb") as infile:
        uploaded = client.files.create(file=infile, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"run": run_dir.name},
    )
    batch_file.write_text(json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id}))
    logger.info("Submitted batch %s (status %s)", batch.id, batch.status)


def get_batch(run_dir: Path, client):
    batch_id = json.loads((run_dir / "batch.json").read_text())["batch_id"]
    return client.batches.retrieve(batch_id)


def collect_run(run_dir: Path, client, model_name: str) -> None:
    batch = get_batch(run_dir, client)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} is {batch.status}, not completed")

    lines = []
    for file_id in [batch.output_file_id, batch.error_file_id]:
        if file_id:
            content = client.files.content(file_id).text
            lines.extend(json.loads(text) for text in content.splitlines() if text.strip())
    results = pd.DataFrame([parse_result_line(line) for line in lines])

    manifest = pd.read_csv(run_dir / "manifest.csv")
    merged = manifest.merge(results, on="custom_id", how="left", validate="one_to_one")
    merged["Response Status"] = merged["Response Status"].fillna("missing")
    merged.insert(0, "Model Name", model_name)
    merged.to_csv(run_dir / "predictions.csv", index=False)
    logger.info("Status counts: %s", merged["Response Status"].value_counts().to_dict())
    logger.info("Wrote %s", run_dir / "predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "submit", "status", "collect"])
    parser.add_argument("--run", required=True, help="Run name; files go in output/<run>/")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--model-name", default="GPT-6 Luna", help="Display name in results")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_dir = OUTPUT_DIR / args.run

    if args.command == "build":
        build_run(run_dir, args.model, args.reasoning_effort, args.max_output_tokens)
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI()
    if args.command == "submit":
        submit_run(run_dir, client)
    elif args.command == "status":
        batch = get_batch(run_dir, client)
        logger.info("Batch %s: %s %s", batch.id, batch.status, batch.request_counts)
    elif args.command == "collect":
        collect_run(run_dir, client, args.model_name)


if __name__ == "__main__":
    main()
