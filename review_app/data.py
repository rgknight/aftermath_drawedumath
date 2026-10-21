"""Turn the hard-question subset (output/subsets, from code/build_subset.py) into review items."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_subset_items(subset_dir: Path) -> dict[str, dict[str, Any]]:
    """Review items keyed by id, each with its model answers and ensemble ratings."""
    questions = pd.read_csv(subset_dir / "hard_teacher_questions.csv")
    explorer = json.loads((subset_dir / "explorer_data.json").read_text())
    model_names = explorer["models"]
    extra_model = explorer.get("extraModel", "Additional model")

    answers: dict[int, dict[str, Any]] = {}
    for path in (subset_dir / "answers").glob("*.json"):
        answers[int(path.stem)] = json.loads(path.read_text())

    items = {}
    for question_id, row in questions.iterrows():
        problem = int(row["Problem ID"])
        payload = answers.get(problem, {"models": {}, "luna": {}})
        models = [
            {"name": name, "answer": answer, "rating": rating}
            for name, (answer, rating) in zip(
                model_names, payload["models"].get(str(question_id), []), strict=False
            )
        ]
        luna = payload["luna"].get(str(question_id))
        if luna:
            models.append({"name": extra_model, "answer": luna[0], "rating": luna[1]})
        items[f"q{question_id:05d}"] = {
            "image": row["Image Name"],
            "image_url": row["Image URL"],
            "problem": problem,
            "question": str(row["Question"]).strip(),
            "answer": str(row["Reference Answer"]).strip(),
            "category": row["Question Cat"],
            "models_correct": int(row["Models correct"]),
            "models": models,
            "status": "pending",
            "source": "subset",
            "order": int(question_id),
        }
    return items
