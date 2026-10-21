"""Back up a run directory (output/<run>/) to a private Hugging Face dataset.

Our generated answers and judge ratings exist nowhere else: OpenAI requests used
store=false and provider batch result files expire. Re-run after each stage to update.

Usage (from the repo root; needs HF_TOKEN and HF_BACKUP_REPO in .env):
    uv run python code/backup_to_hf.py --run luna6_none
"""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

DATASET_CARD = """---
license: cc-by-nc-4.0
---

# Aftermath of DrawEduMath: new-model runs (private)

Answers from newer VLMs on the Aftermath of DrawEduMath matched sample, scored by the
paper's three-judge ensemble. Generated with `code/batch_generate.py` and `code/judge.py`
in the aftermath_drawedumath repo. Questions, reference answers and image URLs come from
[allenai/DrawEduMath](https://huggingface.co/datasets/allenai/DrawEduMath) (CC-BY-NC-4.0).

One folder per run; each has `run_config.json`, `predictions.csv` (model answers),
and once judged, `judges/<judge>/ratings.csv` and `judged.csv`.
"""


def backup_run(run_dir: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError(f"{repo_id} exists and is public; refusing to upload")

    if not api.file_exists(repo_id, "README.md", repo_type="dataset"):
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=run_dir.name,
        commit_message=f"Back up {run_dir.name}",
    )
    logger.info("Uploaded %s to %s (%s)", run_dir, repo_id, commit.commit_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Run name; uploads output/<run>/")
    parser.add_argument("--repo-id", help="Private dataset to upload to (default: HF_BACKUP_REPO)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    repo_id = args.repo_id or os.environ.get("HF_BACKUP_REPO")
    if not repo_id:
        raise SystemExit("Set HF_BACKUP_REPO in .env or pass --repo-id")
    run_dir = OUTPUT_DIR / args.run
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    backup_run(run_dir, repo_id, os.environ["HF_TOKEN"])


if __name__ == "__main__":
    main()
