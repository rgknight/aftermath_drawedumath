"""Compare judged new-model runs with the published models on the Aftermath matched sample.

Follows aftermath.ipynb: images are labeled error / no error from the teacher's answer to
the generic error question (answer_mapping.json), questions are grouped with
questions_to_categories.json, and "Image creation and medium" questions are dropped as in the
redrawing experiment. Reports, per model:
- content-question accuracy on error vs. no-error images (per-image means, t-test), as in
  the notebook's calculate_diff_in_means;
- the same gap controlling for problem (OLS with problem fixed effects);
- accuracy on "Correctness and errors" questions by image type.

Usage (from the repo root):
    uv run python code/compare_runs.py --run luna6_none
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

GENERIC_ERROR_QUESTION = (
    "What errors does the student make in their response? If there are none, write that there "
    "is no error"
)
COLUMNS = [
    "Model Name",
    "Problem ID",
    "Image Name",
    "QA Type",
    "Question",
    "Reference Answer",
    "Score",
]


def map_to_question_category(question: str, question_to_category: dict) -> str:
    question = question.strip()
    if question.startswith("What errors does the student make in their response?"):
        question = "What errors does the student make in their response?"
    return question_to_category[question]


def label_images(predictions: pd.DataFrame, answer_mapping: dict) -> pd.Series:
    """Image Name -> 'error' / 'no error', from the teacher's answer to the generic question."""
    answers = predictions[predictions["Question"] == GENERIC_ERROR_QUESTION]
    answers = answers.groupby("Image Name")["Reference Answer"].first().str.strip()
    return answers.map(lambda answer: "no error" if answer_mapping.get(answer) == "no" else "error")


def load_comparison(run_names: list[str], drop_image_creation: bool = True) -> pd.DataFrame:
    published = pd.read_csv(DATA_DIR / "predictions.csv", usecols=COLUMNS)
    with open(DATA_DIR / "answer_mapping.json") as infile:
        answer_mapping = {answer.strip(): label for answer, label in json.load(infile).items()}
    with open(DATA_DIR / "questions_to_categories.json") as infile:
        question_to_category = json.load(infile)
    with open(DATA_DIR / "sampled_images_per_problem.json") as infile:
        sampled_stems = {Path(name).stem for name in json.load(infile)["1"]}

    image_labels = label_images(published, answer_mapping)
    frames = [published]
    for run_name in run_names:
        frames.append(pd.read_csv(OUTPUT_DIR / run_name / "judged.csv", usecols=COLUMNS))
    combined = pd.concat(frames, ignore_index=True)

    combined = combined[combined["Image Name"].map(lambda name: Path(name).stem in sampled_stems)]
    combined["Has Error"] = combined["Image Name"].map(image_labels)
    combined["Question Cat"] = combined["Question"].map(
        lambda question: map_to_question_category(question, question_to_category)
    )
    if drop_image_creation:
        combined = combined[combined["Question Cat"] != "Image creation and medium"]
    return combined


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from scipy import stats

    rows = []
    for model_name, model_df in combined.groupby("Model Name"):
        content = model_df[model_df["Question Cat"] != "Correctness and errors"]
        per_image = content.groupby(["Problem ID", "Image Name", "Has Error"])["Score"].mean()
        per_image = per_image.reset_index()
        error_scores = per_image.loc[per_image["Has Error"] == "error", "Score"]
        clean_scores = per_image.loc[per_image["Has Error"] == "no error", "Score"]
        t_test = stats.ttest_ind(error_scores, clean_scores)

        per_image["Correct"] = (per_image["Has Error"] == "no error").astype(int)
        per_image = per_image.rename(columns={"Problem ID": "Problem_ID"})
        fixed_effects = smf.ols("Score ~ Correct + C(Problem_ID)", data=per_image).fit()

        correctness = model_df[model_df["Question Cat"] == "Correctness and errors"]
        correctness_by_type = correctness.groupby("Has Error")["Score"].mean()
        rows.append(
            {
                "Model": model_name,
                "Content, error img": error_scores.mean(),
                "Content, no-error img": clean_scores.mean(),
                "Gap": error_scores.mean() - clean_scores.mean(),
                "Gap p": t_test.pvalue,
                "Gap (problem FE)": -fixed_effects.params["Correct"],
                "C&E, error img": correctness_by_type.get("error"),
                "C&E, no-error img": correctness_by_type.get("no error"),
                "Missing scores": int((model_df["Score"] == -1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Content, no-error img", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="Run name(s) in output/")
    args = parser.parse_args()

    combined = load_comparison(args.run)
    summary = summarize(combined)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / args.run[0] / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
