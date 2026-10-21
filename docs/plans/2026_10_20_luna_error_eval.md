# Plan: GPT-6 Luna on the Aftermath matched sample

Goal: see whether a contemporary model still shows the "student error" gap reported in
*Aftermath of DrawEduMath* (models score lower on content questions when the student's
work contains an error).

## Scope (first run)

- **Model:** `gpt-6-luna` (OpenAI; only snapshot is `gpt-6-luna`), `reasoning.effort = "none"`.
- **Questions:** the paper's matched sample (`sampled_images_per_problem.json["1"]`):
  336 original images (168 error / 168 no-error, one of each per problem), 9,136 QA pairs.
  Same rows as every model in `predictions.csv`.
- **Not yet:** redrawn images (exp1), textual-support conditions (exp2), full benchmark.

## Faithfulness to the original pipeline

Mirrors `allenai/DrawEduMath` `scripts/pipeline/generation/generate_openai.py`:

- System/instructions: `GENERATE_ANSWER_PROMPT` (copied verbatim).
- User turn: `"Answer the following question: " + Question` (unstripped), then the image.
- Image detail left at the API default (`auto`), as upstream did.

Deliberate differences, to report alongside results:

| | Paper (GPT-5) | This run |
|---|---|---|
| Endpoint | Chat Completions, synchronous | Responses API via Batch API (50% price) |
| Image | base64 of local `Resized_Merged_Problem_Images` | S3 URL from `DrawEduMath_QA.csv` (`Images_w_Problem_Bodies`) |
| Reasoning effort | model default (`medium` for GPT-5) | `none` |
| Output cap | none | `max_output_tokens = 4096` (matches upstream Anthropic script); incompletes are recorded, not dropped |
| Storage | default | `store: false` |

## Steps

1. `code/batch_generate.py build` — select rows, write `requests.jsonl` + `manifest.csv` to `output/<run>/`.
2. Review the first request by eye; `submit` (one batch, 9,136 requests, 24h window).
3. `collect` — join answers back onto the manifest → `output/<run>/predictions.csv`
   in the pipeline's column format.
4. **Judging (separate step, needs sign-off):** same three-judge ensemble as the paper
   (Claude Sonnet 4.5, Gemini 2.5 Pro, GPT-4o; mode of 1–4 ratings, 3–4 = correct) so
   scores are comparable with `predictions.csv`.
5. Re-run the notebook's "Controlling for Problem" / diff-in-means cells with Luna added.

## Cost (estimate)

~1.2k image tokens + ~0.2k text per request → ~13M input tokens, under the 20M Build-tier
batch queue limit. At batch rates ($0.05/M in, $0.25/M out) ≈ $1 for generation.
Judging cost is separate and larger.
