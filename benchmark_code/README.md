# TutorCaseBench Code

The main pipeline takes pedagogical case-study papers (PDFs) as input and automatically constructs dialogue datasets for evaluating LLM tutors.
Codes are mainly consisted of 3 parts: main data construction pipeline, rubric-based evaluation pipeline, and MCQ generation & evaluation pipeline.

---

## 🏃‍♂️ Before You Start

The following items must be included before running the pipeline.

### 1. PDF Paper Files
```
data/docs/                     # single group
data/docs/{study_group}/       # grouped by study (e.g. data/docs/adhd/0001.pdf)
```
- Place pedagogical case-study PDFs (refer to `document_list.csv`)
- Filenames should follow the `{numericID}.pdf` convention (e.g. `0001.pdf`, `0002.pdf`)
- `run_study_group.py` automatically scans `*.pdf` files in the specified directory

### 2. .env.local — API keys required
See the [.env.local section](#-envlocal).

---

## 📁 Directory Structure

```
project_root/
│
├── .env.local                          # API keys
│
├── main.py                             # Main pipeline (Pipelines 1–5)
├── evaluation_pipeline.py              # Rubric generation + response evaluation
├── quaternary_pipeline.py              # MCQ (4-option) generation + evaluation
├── run_evaluation.py                   # Batch runner for evaluation_pipeline
├── run_study_group.py                  # Full-flow batch runner (per PDF directory)
│
├── config/
│   ├── config_default_model.json       # Default LLM per pipeline stage
│   ├── config_filter.json              # Number of scenarios, category filter threshold
│   ├── config_dialogue_turn.json       # Dialogue turn count range (min/max)
│   ├── config_response.json            # Model list for response collection
│   └── config_evaluation.json          # Evaluation model config, models to test
│
├── prompts/
│   ├── screen_paper.txt
│   ├── analyze_paper.txt
│   ├── extract_excerpt_learner.txt
│   ├── extract_excerpt_instruction.txt
│   ├── build_scenario.txt
│   ├── synthesize_dialogue.txt
│   ├── generate_response.txt
│   ├── generate_rubric.txt
│   ├── evaluate_with_rubric.txt
│   ├── mcq_select_distractors.txt
│   ├── mcq_adapt_options.txt
│   └── mcq_evaluate.txt
│
├── utils/
│   ├── analyze_paper_utils.py          # PDF text extraction
│   ├── prompt_utils.py                 # Prompt building + JSON parsing
│   ├── llm_utils.py                    # Multi-engine LLM calls
│   ├── output_utils.py                 # JSON/text file saving
│   ├── generate_dialogue_utils.py      # Dialogue .txt file generation
│   ├── generate_response_utils.py      # Model response collection
│   └── pipeline_utils.py              # Dependency auto-resolution, path lookup
│
├── data/                               # Runtime-generated
│   ├── docs/                           # Place PDF files here
│   │   └── {study_group}/              # Optional per-group subdirectory
│   ├── logs/
│   ├── pipeline1/
│   │   ├── screen_paper/
│   │   └── analyze_paper/
│   ├── pipeline2/
│   │   ├── extract_excerpt_learner/
│   │   └── extract_excerpt_instruction/
│   ├── pipeline3/
│   │   └── build_scenario/
│   ├── pipeline4/
│   │   ├── synthesize_dialogue/
│   │   └── dialogue/                   # Evaluation .txt files
│   ├── pipeline5/
│   │   └── response/                   # Per-model response .txt files
│   ├── evaluation/
│   │   ├── rubric/
│   │   └── result/
│   └── summaries/
│
└── experiments/
    └── mcq/                            # Runtime-generated
        ├── embeddings_cache_context.json
        ├── embeddings_cache_behavior.json
        ├── embeddings_cache_intervention.json
        └── quaternary/
            ├── questions/
            ├── results/
            └── summary/
```

---

## 🐳 Full Pipeline Flow

```
              ┌─────────────────────────────────────────────────┐
              │                run_study_group.py               │
              │  (Top-level orchestrator for a PDF directory)   │
              └─────────────────────────────────────────────────┘
                                       │ subprocess calls
                    ┌──────────────────▼──────────────────┐
                    │               main.py               │
                    │           Pipeline 1 ~ 5            │
                    └──────────────────┬──────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
     [Pipeline 1]                 [Pipeline 2]                 [Pipeline 3]
  Screen + Analyze Paper      Evidence Extraction            Scenario Construction
  * screen_paper.txt          * extract_excerpt_learner.txt  * build_scenario.txt
  * analyze_paper.txt         * extract_excerpt_instruction.txt
          │                            │                            │
          └────────────────────────────┼────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │            [Pipeline 4]             │
                    │   Dialogue Synthesis + .txt output  │
                    │    * synthesize_dialogue.txt        │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │            [Pipeline 5]             │
                    │  Collect tutor responses per model  │
                    │     * generate_response.txt         │
                    └──────────────────┬──────────────────┘
                                       │
          ┌────────────────────────────┴────────────────────────────┐
          │                                                         │
          ▼                                                         ▼
┌───────────────────────────┐                         ┌──────────────────────────────┐
│  evaluation_pipeline.py   │                         │  quaternary_pipeline.py      │
│  (batch-run via           │                         │  (4-option MCQ)              │
│   run_evaluation.py)      │                         │                              │
│                           │                         │ [--build]                    │
│ [--rubric]                │                         │ * mcq_select_distractors.txt │
│ * generate_rubric.txt     │                         │ * mcq_adapt_options.txt      │
│                           │                         │                              │
│ [--evaluate-all]          │                         │ [--run]                      │
│ * evaluate_with_rubric.txt│                         │ * mcq_evaluate.txt           │
│                           │                         │                              │
│ [--summary-all]           │                         │ [--summary]                  │
└───────────────────────────┘                         └──────────────────────────────┘
```

---

## 📄 Python Files — Details

### 1. `main.py` — Main Dialogue Construction Pipeline

Takes a single PDF paper and runs Pipelines 1–5 sequentially. Each stage is automatically skipped if its output already exists.

```
PDF input
  │
  ├─ [Pipeline 1-A] screen_paper               → data/pipeline1/screen_paper/
  │   * Prompt: screen_paper.txt
  │   * Papers scoring below 3 on any of the 3 dimensions
  │     (student behavior / learning context / teacher actions) are rejected
  │
  ├─ [Pipeline 1-B] analyze_paper              → data/pipeline1/analyze_paper/
  │   * Prompt: analyze_paper.txt
  │   * Extracts student categories (label, demographics, extraction_eligibility, etc.)
  │
  ├─ [Pipeline 2-A] extract_excerpt_learner    → data/pipeline2/extract_excerpt_learner/
  │   * Prompt: extract_excerpt_learner.txt
  │   * Extracts learner evidence (behavior / cognitive / affective / difficulty triggers)
  │     and verbatim student quotes
  │
  ├─ [Pipeline 2-B] extract_excerpt_instruction → data/pipeline2/extract_excerpt_instruction/
  │   * Prompt: extract_excerpt_instruction.txt
  │   * Extracts instructional evidence (immediate/long-term scope, effective/ineffective)
  │
  ├─ [Pipeline 3] build_scenario                → data/pipeline3/build_scenario/
  │   * Prompt: build_scenario.txt
  │   * Builds per-category scenarios (content block + hidden_strategies block)
  │   * Generates student_persona and challenge_categorization
  │
  ├─ [Pipeline 4] synthesize_dialogue           → data/pipeline4/synthesize_dialogue/
  │   * Prompt: synthesize_dialogue.txt         → data/pipeline4/dialogue/*.txt
  │   * Synthesizes a 6-turn dialogue; produces two .txt files per case
  │     (immediate and long_term evaluation prompts)
  │
  └─ [Pipeline 5] collect_model_responses  → data/pipeline5/response/
      * Prompt: generate_response.txt
      * Every model in config_response.json generates a response for each dialogue file
```

**Usage:**
```bash
# Basic run
python main.py data/docs/0001.pdf

# Specify LLM engine
python main.py data/docs/0001.pdf --model openai

# Run only up to a specific pipeline stage, then stop
python main.py data/docs/0001.pdf --pipeline1-only
python main.py data/docs/0001.pdf --pipeline2-only
python main.py data/docs/0001.pdf --pipeline3-only
python main.py data/docs/0001.pdf --pipeline4-only
python main.py data/docs/0001.pdf --pipeline5-only

# Resume from a later stage (reusing existing outputs)
python main.py --pipeline1-output 0001            # reuse analyze_paper output
python main.py --pipeline2-output 0001            # reuse extract_excerpt output
python main.py --pipeline3-output 0001-01         # reuse build_scenario output
python main.py --pipeline4-output 0001-01         # reuse synthesize_dialogue output

# Interactive confirmation before each stage
python main.py data/docs/0001.pdf --check true

# Verbose output
python main.py data/docs/0001.pdf --verbose
```

---

### 2. `evaluation_pipeline.py` — Rubric Generation & Response Evaluation

Evaluates the model responses collected in Pipeline 5 using per-category rubrics.
`immediate` and `long_term` are processed fully independently.

```
[--rubric]
  ├─ IN  data/pipeline3/build_scenario/
  │      data/pipeline4/synthesize_dialogue/
  │      (Prompt: generate_rubric.txt)
  └─ OUT data/evaluation/rubric/rubric_{id}_{ts}.json

[--evaluate / --evaluate-all]
  ├─ IN  data/evaluation/rubric/
  │      data/pipeline5/response/
  │      data/pipeline4/synthesize_dialogue/
  │      (Prompt: evaluate_with_rubric.txt)
  └─ OUT data/evaluation/result/{id}_case_{n}_{type}_{model}_{ts}.json

[--summary-all]
  ├─ IN  data/evaluation/result/
  └─ OUT data/summaries/aggregate_summary_{ts}.json
```

**Usage:**
```bash
# Generate rubric
python evaluation_pipeline.py --rubric --id 0001-01
python evaluation_pipeline.py --rubric --id 0001-01 --model openai

# Evaluate specific files
python evaluation_pipeline.py --evaluate --id 0001-01 \
  --immediate data/pipeline5/response/0001-01_case_001_immediate_gpt-4o.txt
python evaluation_pipeline.py --evaluate --id 0001-01 \
  --long-term data/pipeline5/response/0001-01_case_001_long_term_gpt-4o.txt

# Evaluate all responses for a category
python evaluation_pipeline.py --evaluate-all --id 0001-01
python evaluation_pipeline.py --evaluate-all --id 0001-01 --types immediate
python evaluation_pipeline.py --evaluate-all --id 0001-01 --types immediate long_term

# Parallel workers
python evaluation_pipeline.py --evaluate-all --id 0001-01 --workers 4

# Aggregate summary across all categories
python evaluation_pipeline.py --summary-all
```

---

### 3. `quaternary_pipeline.py` — MCQ (4-Option) Generation & Response Evaluation

Builds, runs, and summarizes 4-option multiple-choice questions. Each question uses one effective intervention from the target scenario as the correct answer, and three effective interventions from other scenarios as distractors. The `--run` stage is where models are evaluated by having them select the best answer.

```
[--build]  Question Construction
  ├─ IN  data/pipeline3/build_scenario/
  │      data/pipeline4/dialogue/
  │      experiments/mcq/embeddings_cache_*   (OpenAI text-embedding-3-large)
  │      (Prompts: mcq_select_distractors.txt
  │                mcq_adapt_options.txt)
  └─ OUT experiments/mcq/quaternary/questions/{question_id}.json

[--run]  Model Evaluation
  ├─ IN  experiments/mcq/quaternary/questions/
  │      config/config_response.json
  │      (Prompt: mcq_evaluate.txt)
  └─ OUT experiments/mcq/quaternary/results/{question_id}_{model}_{ts}.json

[--summary]  Accuracy Aggregation
  ├─ IN  experiments/mcq/quaternary/results/
  └─ OUT experiments/mcq/quaternary/summary/quaternary_summary_{ts}.json
```

**Usage:**
```bash
# Build MCQ questions
python quaternary_pipeline.py --build
python quaternary_pipeline.py --build --sample 50 --workers 8

# Run models on questions (evaluation)
python quaternary_pipeline.py --run
python quaternary_pipeline.py --run --workers 5 --sample 20

# Summarize results
python quaternary_pipeline.py --summary

# Run all three stages at once
python quaternary_pipeline.py --build --run --summary
```

---

### 4. `run_evaluation.py` — Evaluation Batch Runner

Automatically scans `data/pipeline5/response/` for all response files, extracts category IDs from their filenames, and batch-runs `evaluation_pipeline.py` for each category — no need to specify --id manually for every category.

```
data/pipeline5/response/
  │  (auto-collects category IDs from response filenames)
  │
  ├─ Step 1: Generate rubrics       (skipped if already exists)
  ├─ Step 2: Run evaluate-all
  └─ Step 3: Generate aggregate summary
```

**Usage:**
```bash
# Full run (rubric + evaluate + summary)
python run_evaluation.py

# Rubric generation only
python run_evaluation.py --rubric-only

# Evaluation only (rubrics must already exist)
python run_evaluation.py --evaluate-only

# Parallel workers
python run_evaluation.py --workers 4

# Specify response types
python run_evaluation.py --types immediate
python run_evaluation.py --types immediate long_term
```

---

### 5. `run_study_group.py` — Full-Flow Batch Runner

The top-level orchestrator. Takes a PDF directory, runs the full data construction pipeline (`main.py`) on each paper, then automatically moves into rubric generation, response collection, and evaluation (`evaluation_pipeline.py`) — all in one command.

```
data/docs/{study_group}/
  │  (scans all *.pdf files in the given directory)
  │
  ├─ Step 1: main.py (Pipeline 1–5)                — parallelized per paper
  ├─ Step 2: evaluation_pipeline.py --rubric       — per category (skipped if already exists)
  ├─ Step 3: generate_missing_responses            — fill in missing model responses
  └─ Step 4: evaluation_pipeline.py --evaluate-all — full evaluation
```

**Usage:**
```bash
# Full run
python run_study_group.py data/docs/study_group/

# Specify LLM engine
python run_study_group.py data/docs/study_group/ --model openai

# Stop after a specific pipeline
python run_study_group.py data/docs/study_group/ --pipeline1-only
python run_study_group.py data/docs/study_group/ --pipeline2-only
python run_study_group.py data/docs/study_group/ --pipeline3-only
python run_study_group.py data/docs/study_group/ --pipeline4-only
python run_study_group.py data/docs/study_group/ --pipeline5-only

# Start from a later step
python run_study_group.py data/docs/study_group/ --from-rubric     # start from rubric generation
python run_study_group.py data/docs/study_group/ --from-evaluate   # evaluation only

# Parallel workers
python run_study_group.py data/docs/study_group/ --workers 4

# Specify response types
python run_study_group.py data/docs/study_group/ --types immediate

# Skip response generation step
python run_study_group.py data/docs/study_group/ --skip-response-generation
```

---

## 🔧 utils/

| File | Role |
|------|------|
| `analyze_paper_utils.py` | Extracts text from PDFs using PyPDF2. Inserts `#PAGE{n}#` markers per page. Supports `excluded_pages`. |
| `prompt_utils.py` | Loads `prompts/{name}.txt` and substitutes `{KEY}` placeholders. JSON parsing with `json_repair` fallback. |
| `llm_utils.py` | Supports 6 LLM engines: OpenAI / Gemini / Bedrock (Claude) / Vertex AI (Claude) / Vertex OpenAI / Together. Includes retry logic and token usage logging via `LLM_CALL_LOG`. |
| `output_utils.py` | Saves files as `data/{pipeline_name}/{step}_{id}_{timestamp}.json`. |
| `generate_dialogue_utils.py` | Generates `immediate` and `long_term` `.txt` evaluation files from `synthesize_dialogue` output. |
| `generate_response_utils.py` | Collects responses from all models in `config_response.json` for each dialogue file. Skips if file already exists. |
| `pipeline_utils.py` | Parses `paper_id` / `category_id` from `--pipeline{N}-output` arguments and auto-resolves upstream outputs. |

---

## 📝 prompts/

All prompt files use `{KEY}`-style placeholders. System and user prompts are separated by the `<System Prompt/User Prompt>` delimiter.

| File | Pipeline | Key Variables |
|------|----------|---------------|
| `screen_paper.txt` | Pipeline 1-A | `{TEXT}` |
| `analyze_paper.txt` | Pipeline 1-B | `{TEXT}`, `{ID}` |
| `extract_excerpt_learner.txt` | Pipeline 2-A | `{TEXT}`, `{ID}`, `{CATEGORIES}` |
| `extract_excerpt_instruction.txt` | Pipeline 2-B | `{TEXT}`, `{ID}`, `{CATEGORIES}`, `{LEARNER_EXCERPTS}` |
| `build_scenario.txt` | Pipeline 3 | `{ID}`, `{CATEGORY}`, `{CATEGORY_ID}`, `{LEARNER_EXCERPT}`, `{INSTRUCTION_EXCERPT}`, `{NUM_SCENARIOS}` |
| `synthesize_dialogue.txt` | Pipeline 4 | `{ID}`, `{CATEGORY}`, `{CATEGORY_ID}`, `{SCENARIO}`, `{STUDENT_PERSONA}`, `{EVIDENCE}`, `{TURN}` |
| `generate_response.txt` | Pipeline 5 | `{PROMPT}` |
| `generate_rubric.txt` | Rubric Generation | `{STUDENT_DESCRIPTION}`, `{BEHAVIORAL_PATTERNS}`, `{UNIQUE_TRIGGERS_AND_NEEDS}`, `{APPROACH_EFFECTIVENESS}`, `{PRIMARY_CHALLENGE}`, `{EFFECTIVE_IMMEDIATE}`, `{EFFECTIVE_LONG_TERM}`, `{INEFFECTIVE}`, `{DIALOGUE_EXAMPLES}`, `{RUBRIC_TYPES}` |
| `evaluate_with_rubric.txt` | Rubric Evaluation | `{RUBRIC}`, `{DIALOGUE}`, `{FULL_CONTEXT}`, `{STUDENT_DESCRIPTION}`, `{TUTOR_RESPONSE}` |
| `mcq_select_distractors.txt` | MCQ Build | `{DIALOGUE}`, `{STUDENT_PERSONA}`, `{TARGET_CONTEXT}`, `{RESPONSE_TYPE}`, `{CORRECT_INTERVENTION}`, `{SIBLING_INTERVENTIONS}`, `{CANDIDATE_DISTRACTORS}` |
| `mcq_adapt_options.txt` | MCQ Build | `{DIALOGUE}`, `{TARGET_CONTEXT}`, `{STUDENT_PERSONA}`, `{RESPONSE_TYPE}`, `{CORRECT_INSTRUCTION}`, `{DISTRACTOR_1_INSTRUCTION}`, `{DISTRACTOR_2_INSTRUCTION}`, `{DISTRACTOR_3_INSTRUCTION}` |
| `mcq_evaluate.txt` | MCQ Response Generation | `{DIALOGUE}`, `{OPTIONS}` |

---

## ⚙️ config/

### `config_default_model.json`
Specifies the default LLM for each pipeline stage. The models below reflect the configuration used in our experiments.
```json
{
  "pipeline1": { "model_type": "gemini", "model_name": "gemini-3-flash-preview" },
  "pipeline2": { "model_type": "gemini", "model_name": "gemini-3-flash-preview" },
  "pipeline3": { "model_type": "gemini", "model_name": "gemini-3.1-pro-preview" },
  "pipeline4": { "model_type": "gemini", "model_name": "gemini-3.1-pro-preview" }
}
```

### `config_filter.json`
Sets the eligibility thresholds applied at each stage of the pipeline and the target number of scenarios to generate per category.

* `screen_paper`: A paper is rejected outright if any of the three dimensions (student behavior, learning context, teacher actions) scores below 3 on a 1–5 scale.
* `pipeline1_analyze_paper`: Within a passing paper, individual student categories are dropped if any dimension scores 3 or below — a stricter cutoff than screening, since downstream dialogue synthesis requires per-category evidence to be concrete on its own.
* `pipeline2_extract_excerpt`: A category is further filtered out if the LLM extracts zero learner excerpts or zero instructional excerpts, as a grounded scenario requires at least one span of each kind.
* `scenario.num_scenarios`: The recommended number of scenarios to generate per category in Pipeline 3. Passed as a guideline to the LLM and not strictly enforced — actual output may vary depending on available evidence.
```json
{
  "filters": {
    "screen_paper": {
      "score_min": 3,
      "dims": ["student_behavior", "learning_context", "teacher_actions"],
      "description": "Paper rejected if any required dim score < 3 (score 1-5 scale)"
    },
    "pipeline1_analyze_paper": {
      "score_min": 4,
      "dims": ["student_behavior", "learning_context", "teacher_actions"],
      "description": "Category ineligible if any required dim score <= 3"
    },
    "pipeline2_extract_excerpt": {
      "excerpts_min": 1,
      "description": "Category filtered if learner_count = 0 OR instruction_count = 0"
    }
  },
  "scenario": {
    "num_scenarios": 3,
    "description": "Recommended number of scenarios per student category in Pipeline 3 (not enforced)"
  }
}
```

### `config_dialogue_turn.json`
Sets the dialogue turn count range. The actual number of turns is sampled randomly between min and max.
```json
{
  "min_turn": 3,
  "max_turn": 3
}
```

### `config_response.json`
Defines the models used for response collection in Pipeline 5 and MCQ `--run`. The list below includes models evaluated in our experiments.
```json
{
  "models_to_test": [
    { "name": "claude-opus-4-7",                                    "model_type": "vertex",       "description": "Claude Opus 4.7 (Vertex)" },
    { "name": "claude-haiku-4-5@20251001",                          "model_type": "vertex",       "description": "Claude Haiku 4.5 (Vertex)" },
    { "name": "gemini-3.1-pro-preview",                             "model_type": "gemini",       "description": "Gemini 3.1 Pro" },
    { "name": "gemini-3-flash-preview",                             "model_type": "gemini",       "description": "Gemini 3.0 Flash" },
    { "name": "gpt-5.5",                                            "model_type": "openai",       "description": "GPT-5.5" },
    { "name": "gpt-5.4-mini",                                       "model_type": "openai",       "description": "GPT-5.4 Mini" },
    { "name": "meta/llama-4-maverick-17b-128e-instruct-maas",       "model_type": "vertex_openai","description": "Llama 4 Maverick (Vertex MaaS, us-east5)" },
    { "name": "gemma-4-31b-it",                                     "model_type": "gemini",       "description": "Gemma 4 31B Instruct (AI Studio)" },
    { "name": "deepseek-ai/DeepSeek-V4-Pro",                        "model_type": "together",     "description": "DeepSeek V4 Pro (Together)" },
    { "name": "openai/gpt-oss-120b",                                "model_type": "together",     "description": "GPT-OSS 120B (Together)" },
    { "name": "Qwen/Qwen3.5-397B-A17B",                             "model_type": "together",     "description": "Qwen 3.5 397B MoE (Together)" }
  ],
  "output_dirs": {
    "dialogue": "data/pipeline4/dialogue",
    "response": "data/pipeline5/response",
    "assessment": "data/evaluation/result"
  }
}
```

### `config_evaluation.json`
Configures the rubric generation model, the evaluation model, and the list of models whose responses will be evaluated.
```json
{
  "rubric_model": {
    "model_type": "gemini",
    "model_name": "gemini-3.1-pro-preview"
  },
  "evaluation_model": {
    "model_type": "gemini",
    "model_name": "gemini-3.1-pro-preview"
  },
  "models_to_test": [
    { "name": "claude-opus-4-7",                                    "model_type": "vertex",       "description": "Claude Opus 4.7 (Vertex)" },
    { "name": "claude-haiku-4-5@20251001",                          "model_type": "vertex",       "description": "Claude Haiku 4.5 (Vertex)" },
    { "name": "gemini-3.1-pro-preview",                             "model_type": "gemini",       "description": "Gemini 3.1 Pro" },
    { "name": "gemini-3-flash-preview",                             "model_type": "gemini",       "description": "Gemini 3.0 Flash" },
    { "name": "gpt-5.5",                                            "model_type": "openai",       "description": "GPT-5.5" },
    { "name": "gpt-5.4-mini",                                       "model_type": "openai",       "description": "GPT-5.4 Mini" },
    { "name": "meta/llama-4-maverick-17b-128e-instruct-maas",       "model_type": "vertex_openai","description": "Llama 4 Maverick (Vertex MaaS, us-east5)" },
    { "name": "gemma-4-31b-it",                                     "model_type": "gemini",       "description": "Gemma 4 31B Instruct (AI Studio)" },
    { "name": "deepseek-ai/DeepSeek-V4-Pro",                        "model_type": "together",     "description": "DeepSeek V4 Pro (Together)" },
    { "name": "openai/gpt-oss-120b",                                "model_type": "together",     "description": "GPT-OSS 120B (Together)" },
    { "name": "Qwen/Qwen3.5-397B-A17B",                             "model_type": "together",     "description": "Qwen 3.5 397B MoE (Together)" }
  ]
}
```

---

## 🔑 .env.local

Set only the keys corresponding to the LLM engines you intend to use.

```env
# ── OpenAI (model_type: "openai") ──────────────────────────────
OPENAI_API_KEY=sk-...

# ── Google Gemini (model_type: "gemini") ───────────────────────
GEMINI_API_KEY=...
# or
GOOGLE_API_KEY=...

# ── AWS Bedrock / Claude (model_type: "anthropic") ─────────────
AWS_ACCESS_KEY=...                  # or AWS_ACCESS_KEY_ID
AWS_SECRET_KEY=...                  # or AWS_SECRET_ACCESS_KEY
AWS_REGION=us-east-1                # or AWS_DEFAULT_REGION

# ── GCP Vertex AI / Claude (model_type: "vertex") ──────────────
GOOGLE_CLOUD_PROJECT=...            # or GCP_PROJECT_ID
VERTEX_REGION=us-east5              # or GOOGLE_CLOUD_REGION
# ADC auth: gcloud auth application-default login
# or: GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# ── GCP Vertex AI / OpenAI-compatible (model_type: "vertex_openai") ──
# For Llama, Qwen, Mistral, etc. served via Vertex AI Partner MaaS
GOOGLE_CLOUD_PROJECT=...            # shared with above
VERTEX_OAI_REGION=us-east5          # region varies by model (Llama 4: us-east5)

# ── Together AI (model_type: "together") ───────────────────────
TOGETHER_API_KEY=...

# ── LLM call log (optional — logs tokens/latency as JSONL when set) ──
LLM_CALL_LOG=data/logs/llm_usage.jsonl
```

---

## 📦 Installation

```bash
pip install PyPDF2 json-repair python-dotenv openai google-genai anthropic boto3 google-auth
```

Engine-specific optional installs:
```bash
pip install "anthropic[vertex]"              # for Vertex AI Claude
pip install boto3                             # for AWS Bedrock
pip install google-auth google-auth-httplib2  # for Vertex OpenAI (partner MaaS)
```

---

## 💡 Quick Start

```bash
# 1. Set up environment
cp env.local .env.local    # fill in API keys

# 2. Place PDFs
mkdir -p data/docs
cp your_papers/*.pdf data/docs/

# 3. Run the full pipeline on a single paper
python main.py data/docs/0001.pdf --verbose

# 4. Batch run on a directory
python run_study_group.py data/docs/study_group/ --workers 2

# 5. Evaluation only (after response collection is complete)
python run_evaluation.py --workers 2

# 6. Build and run MCQ
python quaternary_pipeline.py --build --workers 8
python quaternary_pipeline.py --run --workers 5
python quaternary_pipeline.py --summary
```