# Question review app

Teachers review the hard-question subset (`code/build_subset.py`): for each question they see the
student work, the question, the teacher's answer and every model's answer with its rating, then
approve, reject (one-line reason), modify, or skip the image. Design and decisions:
`docs/plans/2026_10_21_teacher_review_app.md`.

How it behaves:
- Reviewers sign in with a plain ID you've added. The ID is the only gate: anyone who knows an
  ID can review as that person.
- A reviewer holds a whole image, hardest first, for 60 minutes after their last action.
- Modify closes the question and queues the edited version (no model answers) for another reviewer.
- Skip hands the image back; the person who skipped isn't offered it again.

## Try it locally (no GCP)

Needs `output/subsets/` from `uv run python code/build_subset.py`. Data lives in memory and is
lost when you stop the server.

```
REVIEW_STORE=memory DEV_ANNOTATORS=ryan,test.teacher \
  uv run uvicorn review_app.app:create_app_from_env --factory --port 8766
```

Open http://127.0.0.1:8766 and sign in as `ryan`.

## One-time GCP setup

Copy `.env.deploy.example` to `.env.deploy` (gitignored) and fill in `GCP_PROJECT` and
`GCP_REGION`. Then, with `gcloud` signed in to an account that owns the project:

```
set -a; source .env.deploy; set +a

gcloud services enable run.googleapis.com firestore.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com --project "$GCP_PROJECT"

# Firestore in native mode, plus the one index the queue query needs
gcloud firestore databases create --location="$GCP_REGION" --project "$GCP_PROJECT"
gcloud firestore indexes composite create --collection-group=images \
  --field-config=field-path=open,order=ascending \
  --field-config=field-path=priority,order=ascending --project "$GCP_PROJECT"

# Session-signing secret (never written to disk)
python3 -c "import secrets; print(secrets.token_urlsafe(32), end='')" | \
  gcloud secrets create "$SESSION_SECRET_NAME" --data-file=- --project "$GCP_PROJECT"

# Let Cloud Run's service account use Firestore and read the secret
SA="$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$GCP_PROJECT" --member="serviceAccount:$SA" --role=roles/datastore.user
gcloud secrets add-iam-policy-binding "$SESSION_SECRET_NAME" --member="serviceAccount:$SA" \
  --role=roles/secretmanager.secretAccessor --project "$GCP_PROJECT"

# Local credentials for the manage.py commands below
gcloud auth application-default login
```

## Load the queue and add reviewers

```
uv run python code/build_subset.py                    # if output/subsets/ is missing
uv run python -m review_app.manage seed               # refuses if Firestore already has items
uv run python -m review_app.manage add-reviewers maria.lopez jamal.k
```

## Deploy

```
review_app/deploy.sh
```

`gcloud` prints the service URL when it finishes. Share it with reviewers directly; don't put it
in this repo.

## While reviews run

```
uv run python -m review_app.manage status             # counts by status, reviews per person
uv run python -m review_app.manage export             # items.csv + reviews.csv in output/review_exports/
uv run python code/backup_to_hf.py --run review_exports
```

`reviews.csv` has one row per action (approve, reject with reason, modify with the new text,
skip), with the reviewer and a Unix timestamp. `items.csv` has every question's final status,
including edited versions and the item they replaced (`parent`).
