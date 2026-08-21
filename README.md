# Week 10 — LLMOps with Gemini Fine-Tuning on IRIS

This implementation prepares two representations of the IRIS dataset, fine-tunes the same Gemini model on each representation, evaluates both tuned models on the same held-out test set, and optionally gates GitHub Actions on evaluation thresholds.

## Repository structure

```text
.
├── data/
│   ├── iris.csv                 # add this yourself
│   └── README.txt
├── scripts/
│   ├── split_data.py
│   ├── prepare_data.py
│   ├── prepare_all.py
│   ├── tune.py
│   ├── evaluate.py
│   └── compare_results.py
├── .github/workflows/week10-evaluation.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Experiment design

Only the **data representation** changes.

v1 input:
`sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2`

v1 output:
`setosa`

v2 input:
`A flower specimen has a sepal length of 5.1 cm, sepal width of 3.5 cm, petal length of 1.4 cm, and petal width of 0.2 cm. Identify the iris species.`

v2 output:
`This is Iris setosa.`

Both versions use the same deterministic train/validation/test split, base model, epochs, learning-rate multiplier, and adapter size.

## 2. Current Vertex AI model choice

The default is `gemini-2.5-flash`. Google currently lists Gemini 2.5 Flash, Gemini 2.5 Flash-Lite, Gemini 2.5 Pro, Gemini 3.1 Flash-Lite, and Gemini 3.5 Flash as supporting supervised tuning. Check your project's region/model availability before submission.

## 3. Authentication

For Vertex AI Workbench/local development:

```bash
gcloud auth login
gcloud auth application-default login
```

Then:

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GCS_BUCKET="YOUR_BUCKET"
```

Do not commit credentials.

## 4. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Add IRIS data

Put the CSV at:

```text
data/iris.csv
```

Required columns:

```text
sepal_length,sepal_width,petal_length,petal_width,species
```

## 6. Split data

```bash
python scripts/split_data.py
```

This creates a stratified 80/16/20? **No** — the implementation uses 20% test first and then 20% of the remaining 80% as validation, giving approximately:

```text
Train       64%
Validation  16%
Test        20%
```

With the standard 150-row Iris dataset this is approximately 96/24/30.

The same `test.csv` is used for v1 and v2.

## 7. Generate JSONL

```bash
python scripts/prepare_all.py
```

Creates:

```text
data/iris_v1_train.jsonl
data/iris_v1_validation.jsonl
data/iris_v2_train.jsonl
data/iris_v2_validation.jsonl
```

Inspect:

```bash
head -n 3 data/iris_v1_train.jsonl
head -n 3 data/iris_v2_train.jsonl
```

## 8. Upload to GCS

```bash
gcloud storage mkdir gs://$GCS_BUCKET/week10

gcloud storage cp data/iris_v1_train.jsonl \
  gs://$GCS_BUCKET/week10/iris_v1_train.jsonl

gcloud storage cp data/iris_v1_validation.jsonl \
  gs://$GCS_BUCKET/week10/iris_v1_validation.jsonl

gcloud storage cp data/iris_v2_train.jsonl \
  gs://$GCS_BUCKET/week10/iris_v2_train.jsonl

gcloud storage cp data/iris_v2_validation.jsonl \
  gs://$GCS_BUCKET/week10/iris_v2_validation.jsonl

gcloud storage ls gs://$GCS_BUCKET/week10/
```

## 9. Fine-tuning hyperparameters

The two jobs use:

```text
Base model:                gemini-2.5-flash
Epochs:                    5
Learning-rate multiplier:  1.0
Adapter size:              8
```

Keep these identical for both jobs.

## 10. Fine-tune v1

```bash
python scripts/tune.py \
  --version v1 \
  --train-gcs "gs://$GCS_BUCKET/week10/iris_v1_train.jsonl" \
  --validation-gcs "gs://$GCS_BUCKET/week10/iris_v1_validation.jsonl" \
  --output-gcs "gs://$GCS_BUCKET/week10/tuning/v1" \
  --display-name "week10-iris-v1-raw" \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location "$GOOGLE_CLOUD_LOCATION" \
  --base-model "gemini-2.5-flash" \
  --epochs 5 \
  --learning-rate-multiplier 1.0 \
  --adapter-size 8
```

Save the printed tuned endpoint.

## 11. Fine-tune v2

```bash
python scripts/tune.py \
  --version v2 \
  --train-gcs "gs://$GCS_BUCKET/week10/iris_v2_train.jsonl" \
  --validation-gcs "gs://$GCS_BUCKET/week10/iris_v2_validation.jsonl" \
  --output-gcs "gs://$GCS_BUCKET/week10/tuning/v2" \
  --display-name "week10-iris-v2-natural-language" \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location "$GOOGLE_CLOUD_LOCATION" \
  --base-model "gemini-2.5-flash" \
  --epochs 5 \
  --learning-rate-multiplier 1.0 \
  --adapter-size 8
```

## 12. REST fine-tuning alternative

For v1:

```bash
PROJECT_ID="$GOOGLE_CLOUD_PROJECT"
LOCATION="$GOOGLE_CLOUD_LOCATION"

curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/tuningJobs" \
  -d "{
    \"baseModel\": \"gemini-2.5-flash\",
    \"supervisedTuningSpec\": {
      \"trainingDatasetUri\": \"gs://${GCS_BUCKET}/week10/iris_v1_train.jsonl\",
      \"validationDatasetUri\": \"gs://${GCS_BUCKET}/week10/iris_v1_validation.jsonl\",
      \"hyperParameters\": {
        \"epochCount\": 5,
        \"learningRateMultiplier\": 1.0,
        \"adapterSize\": 8
      }
    },
    \"tunedModelDisplayName\": \"week10-iris-v1-raw\"
  }"
```

For v2, change only the two dataset URIs and display name.

## 13. Evaluate v1

```bash
python scripts/evaluate.py \
  --test data/test.csv \
  --model "V1_TUNED_MODEL_ENDPOINT" \
  --version v1 \
  --output results/v1.json
```

## 14. Evaluate v2

```bash
python scripts/evaluate.py \
  --test data/test.csv \
  --model "V2_TUNED_MODEL_ENDPOINT" \
  --version v2 \
  --output results/v2.json
```

The evaluator records:

- accuracy
- per-class precision
- per-class recall
- per-class F1
- format compliance
- raw response
- parsed prediction
- correctness

### Format compliance

v1 is compliant only when the response is exactly:

```text
setosa
versicolor
virginica
```

v2 is compliant only when the response is exactly:

```text
This is Iris setosa.
This is Iris versicolor.
This is Iris virginica.
```

These are deliberately non-compliant:

```text
Setosa
The answer is setosa.
{"species":"setosa"}
This is iris setosa.
```

## 15. Compare

```bash
python scripts/compare_results.py \
  --v1 results/v1.json \
  --v2 results/v2.json
```

Do not claim which representation is better until the actual tuned models have been evaluated.

## 16. GitHub Actions

Configure these repository secrets:

```text
GCP_PROJECT_ID
GCP_SERVICE_ACCOUNT_KEY
V1_TUNED_MODEL_ENDPOINT
V2_TUNED_MODEL_ENDPOINT
```

The workflow runs on pushes to `week_10` and fails when either tuned model falls below:

```text
accuracy >= 0.80
format compliance >= 0.80
```

For production, prefer GitHub Actions Workload Identity Federation instead of long-lived service-account keys.

## 17. Git commands

```bash
git checkout -b week_10
git add .
git commit -m "Implement Week 10 LLMOps Gemini fine-tuning"
git push -u origin week_10
```

## 18. Screencast order

Show:

1. `week_10` branch.
2. Repository structure.
3. v1 JSONL.
4. v2 JSONL.
5. GCS objects.
6. Vertex AI v1 tuning job.
7. Vertex AI v2 tuning job.
8. Same model/hyperparameters.
9. Held-out test set.
10. v1 metrics.
11. v2 metrics.
12. Metric comparison.
13. Explanation of the representation difference.
14. GitHub Actions result if Task 5 is included.

## 19. Final comparison template

After running the experiment, fill in:

```text
v1 accuracy: __________
v2 accuracy: __________

v1 format compliance: __________
v2 format compliance: __________

Best representation: __________

Explanation:
The better-performing representation was __________.
A possible reason is __________________________________________.
```

Do not invent metrics before running the jobs.
