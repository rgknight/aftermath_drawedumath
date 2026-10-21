# Plan: teacher review app for the hard-question subset

Goal: teachers review the 4,983 hard teacher-written QA pairs (`code/build_subset.py`) and
approve, reject (with a one-line reason) or modify each one, so we can separate genuinely
hard questions from noisy ones (ambiguous questions, wrong or debatable reference answers).

## Requirements (from Ryan)

- Log in with a pre-issued ID; no ID, no access.
- One screen: image, question, teacher answer, model answers and ratings.
- Actions: Approve, Reject (one-line reason required), Modify. Then go straight to the next item.
- No browsing, search or filters.
- Lock an image while someone reviews it, so two annotators never work the same image.
- A modified question goes back into the pool as a new item, without model answers or ratings.
- Queue runs hardest to easiest (fewest of the 11 models correct first).
- Deployed on GCP; the deployment URL and other deployment details stay out of this public repo.

## Proposed design

**Stack:** one Cloud Run service (Python, FastAPI, server-rendered HTML; no front-end build)
plus Firestore (native mode). Both scale to zero, so idle cost is close to nothing, and
Firestore transactions give us safe locks across Cloud Run instances. Cloud SQL would also
work (`SELECT … FOR UPDATE SKIP LOCKED`) but costs money while idle.

**Data (Firestore):**
- `annotators/{sha256(id)}`: display label, active flag. Raw IDs are never stored.
- `items/{item_id}`: image, problem, question, teacher answer, category, models correct
  (null for modified items), source (`subset` / `modified`), parent item, version, status
  (`pending` / `approved` / `rejected` / `modified`), queue priority.
- `images/{image_name}`: lock holder, lock expiry.
- `reviews/{auto_id}`: item, annotator, decision, reason, edited question/answer,
  shown-at and submitted-at times. Append-only; the audit trail and the export source.
- Model answers and ratings: the per-problem answer files from `build_subset.py`, loaded
  into Firestore or bundled with the image. Not committed (public repo).

**Queue and locking:**
- An annotator claims the next unlocked image in a transaction: the image whose hardest
  pending item has the fewest models correct. They review that image's pending items
  hardest-first, then claim the next image.
- Locks are leases (e.g. 20 minutes), renewed while the page is open and released on
  submit or logout. An abandoned lock expires and the image returns to the pool.
- Modify closes the original item as `modified` and creates a new pending item (no model
  answers) with a link to its parent. Proposed: modified items go to the front of the queue
  and can't be reviewed by the annotator who wrote them.

**Login:** plain, human-readable IDs (e.g. `maria.lopez`) so we always know who annotated what.
Sign-in only checks the ID is on the list. A signed session cookie (secret in Secret Manager);
HTTPS comes with Cloud Run.

**Keeping deployment details out of the repo:** project ID, region, service name and URL live
in a gitignored `.env.deploy`; the deploy script reads them from the environment. Seed data,
annotator IDs and exports never enter the repo.

**Export:** a script writes all reviews and the final item states to CSV, and backs them up to
the private Hugging Face dataset with `code/backup_to_hf.py`.

## Decisions (2026-10-21)

1. Single review per item; see how annotators do before adding double review.
2. Show model names, answers and ratings; their value outweighs anchoring risk.
3. Modify can change the question and the teacher answer.
4. Include the standard "What errors does the student make?" question; get through what we can.
5. Skip returns the image to the queue; the person who skipped isn't offered it again.
6. Modified items keep their parent's place in the queue. Their author isn't served them
   (otherwise Modify would immediately show you your own edit); anyone else can review them.
7. Lock lasts 60 minutes from the reviewer's last action.
8. A guidelines pop-up explains the project and when to approve, reject, modify or skip.
9. No re-running models on modified questions.
10. Data is de-identified and covered by prior research permissions.
