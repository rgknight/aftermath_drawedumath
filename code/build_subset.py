"""Build the hard teacher-question subset of DrawEduMath and the data for its explorer.

Keeps questions that are
- written by teachers (drops the Claude- and GPT-4o-generated QA pairs),
- not about how the image was made ("Image creation and medium"), and
- answered correctly by fewer than half of the 11 published models (5 or fewer),
over the full benchmark (2,030 images). Questions any model lacks a usable score for are dropped.

Usage (from the repo root):
    uv run python code/build_subset.py
Writes output/subsets/hard_teacher_questions.csv and output/subsets/explorer_data.json.
"""

import json
import logging

import pandas as pd
from compare_runs import DATA_DIR, OUTPUT_DIR, label_images, map_to_question_category

logger = logging.getLogger(__name__)

SUBSET_DIR = OUTPUT_DIR / "subsets"
QUESTION_KEY = ["Problem ID", "Image Name", "QA Type", "Question", "Reference Answer"]
# Order used in the paper's figures: by developer, then release.
MODEL_ORDER = [
    "Gemini 2.0 Flash",
    "Gemini 2.5 Pro Preview",
    "Gemini 2.5 Pro",
    "GPT-4.1",
    "GPT-4.5 Preview",
    "o4 mini",
    "GPT-5",
    "Claude Sonnet 3.7",
    "Claude Sonnet 4",
    "Claude Sonnet 4.5",
    "Llama 4 Scout",
]
# The new-model run shown alongside the published models where it covers a question.
LUNA_RUN = "luna6_none"
LUNA_JUDGES = ["Claude", "Gemini", "Openai"]


def select_hard_teacher_questions(questions: pd.DataFrame, model_count: int) -> pd.DataFrame:
    """Teacher-written, non image/medium questions that most models get wrong."""
    keep = (
        (questions["QA Type"] == "teacher")
        & (questions["Question Cat"] != "Image creation and medium")
        & (questions["Models scored"] == model_count)
        & (questions["Models correct"] <= model_count // 2)
    )
    return questions[keep].reset_index(drop=True)


def score_questions(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per question: how many models were scored, how many were correct, and which."""
    scored = predictions[predictions["Score"].isin([0, 1])]
    per_model = scored.pivot_table(index=QUESTION_KEY, columns="Model Name", values="Score")
    per_model = per_model.reindex(columns=MODEL_ORDER)
    questions = per_model.notna().sum(axis=1).rename("Models scored").to_frame()
    questions["Models correct"] = per_model.sum(axis=1).astype(int)
    questions["Correct by"] = per_model.apply(
        lambda row: "".join("1" if score == 1 else "0" for score in row), axis=1
    )
    return questions.reset_index()


def collect_answers(
    subset: pd.DataFrame, predictions: pd.DataFrame, luna: pd.DataFrame, model_order: list[str]
) -> dict[int, dict]:
    """Per problem: each subset question's model answers and ratings, keyed by question id.

    "models" holds [answer, ensemble rating] per model in model_order ("" and -1 if missing);
    "luna" holds [answer, ensemble, [claude, gemini, gpt-4o ratings], [their reasons]] for the
    questions the Luna run covered. The question id is the row's index in the subset.
    """
    ids = subset[QUESTION_KEY].assign(question_id=subset.index)
    answers = predictions.merge(ids, on=QUESTION_KEY)[
        ["question_id", "Model Name", "Model Answer", "Ensemble_Judge_Rating"]
    ]
    lookup = {
        (question_id, model): ["" if pd.isna(answer) else str(answer), int(rating)]
        for question_id, model, answer, rating in answers.itertuples(index=False)
    }
    files: dict[int, dict] = {}
    for question_id, row in subset.iterrows():
        problem = files.setdefault(int(row["Problem ID"]), {"models": {}, "luna": {}})
        problem["models"][str(question_id)] = [
            lookup.get((question_id, model), ["", -1]) for model in model_order
        ]
    for _, row in luna.merge(ids, on=QUESTION_KEY).iterrows():
        files[int(row["Problem ID"])]["luna"][str(row["question_id"])] = [
            "" if pd.isna(row["Model Answer"]) else str(row["Model Answer"]),
            int(row["Ensemble_Judge_Rating"]),
            [int(row[f"{judge}_Judge_Rating"]) for judge in LUNA_JUDGES],
            [str(row[f"{judge}_Judge_Reason"]) for judge in LUNA_JUDGES],
        ]
    return files


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    predictions = pd.read_csv(DATA_DIR / "predictions.csv")
    with open(DATA_DIR / "answer_mapping.json") as infile:
        answer_mapping = {answer.strip(): label for answer, label in json.load(infile).items()}
    with open(DATA_DIR / "questions_to_categories.json") as infile:
        question_to_category = json.load(infile)
    qa = pd.read_csv(DATA_DIR / "DrawEduMath_QA.csv").set_index("Image Name")

    questions = score_questions(predictions)
    questions["Question Cat"] = questions["Question"].map(
        lambda question: map_to_question_category(question, question_to_category)
    )
    subset = select_hard_teacher_questions(questions, model_count=len(MODEL_ORDER))
    subset["Has Error"] = subset["Image Name"].map(label_images(predictions, answer_mapping))
    subset["Image URL"] = subset["Image Name"].map(qa["Image URL"])

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(SUBSET_DIR / "hard_teacher_questions.csv", index=False)
    logger.info(
        "Kept %s questions on %s images (%s problems)",
        len(subset),
        subset["Image Name"].nunique(),
        subset["Problem ID"].nunique(),
    )

    luna = pd.read_csv(OUTPUT_DIR / LUNA_RUN / "judged.csv")
    answers_dir = SUBSET_DIR / "answers"
    answers_dir.mkdir(exist_ok=True)
    for old_file in answers_dir.glob("*.json"):
        old_file.unlink()
    files = collect_answers(subset, predictions, luna, MODEL_ORDER)
    for problem, payload in files.items():
        (answers_dir / f"{problem}.json").write_text(json.dumps(payload, separators=(",", ":")))
    luna_question_ids = {int(qid) for payload in files.values() for qid in payload["luna"]}
    logger.info(
        "Wrote answers for %s problems (%.1f MB); Luna covers %s questions",
        len(files),
        sum(path.stat().st_size for path in answers_dir.glob("*.json")) / 1e6,
        len(luna_question_ids),
    )

    images = sorted(subset["Image Name"].unique())
    image_index = {name: index for index, name in enumerate(images)}
    explorer = {
        "models": MODEL_ORDER,
        "extraModel": "GPT-6 Luna",
        "categories": sorted(subset["Question Cat"].unique()),
        "images": [
            {
                "name": name,
                "url": qa.at[name, "Image URL"],
                "problem": int(qa.at[name, "Problem ID"]),
                "caption": str(qa.at[name, "Image Caption"]).strip(),
                "error": subset.loc[subset["Image Name"] == name, "Has Error"].iloc[0] == "error",
            }
            for name in images
        ],
        "questions": [
            [
                image_index[row["Image Name"]],
                row["Question"].strip(),
                row["Reference Answer"].strip(),
                row["Question Cat"],
                row["Correct by"],
                int(question_id in luna_question_ids),
            ]
            for question_id, row in subset.iterrows()
        ],
    }
    (SUBSET_DIR / "explorer_data.json").write_text(json.dumps(explorer, separators=(",", ":")))
    logger.info("Explorer data: %.1f MB", (SUBSET_DIR / "explorer_data.json").stat().st_size / 1e6)


if __name__ == "__main__":
    main()
