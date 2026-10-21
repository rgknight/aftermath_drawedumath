#!/usr/bin/env bash
# Deploy the review app to Cloud Run. Project details come from .env.deploy (gitignored),
# so nothing about the deployment is stored in this public repo. See docs/review_app.md.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env.deploy ]]; then
  echo "Missing .env.deploy at the repo root; copy .env.deploy.example and fill it in." >&2
  exit 1
fi
set -a
source .env.deploy
set +a

: "${GCP_PROJECT:?Set GCP_PROJECT in .env.deploy}"
: "${GCP_REGION:?Set GCP_REGION in .env.deploy}"
SERVICE="${REVIEW_SERVICE:-question-review}"
SECRET="${SESSION_SECRET_NAME:-review-session-secret}"

gcloud run deploy "$SERVICE" \
  --project "$GCP_PROJECT" \
  --region "$GCP_REGION" \
  --source review_app \
  --allow-unauthenticated \
  --max-instances 3 \
  --memory 512Mi \
  --set-env-vars "GCP_PROJECT=$GCP_PROJECT,FIRESTORE_DATABASE=${FIRESTORE_DATABASE:-(default)}" \
  --set-secrets "SESSION_SECRET=$SECRET:latest"
