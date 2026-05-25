# TutorCaseBench

**TutorCaseBench** is a benchmark for evaluating whether language models can identify appropriate pedagogical interventions for students with learning and behavioral difficulties. Each item is grounded in real case studies from pedagogy, educational psychology, and special education, and tests models across two complementary formats: open-ended response generation and multiple-choice intervention selection (MCQ).

---

## 🔎 Overview

The benchmark is organized around three core units: **categories**, **scenarios**, and **rubrics**.

- A **category** corresponds to a distinct student profile drawn from a source paper. All scenarios within a category share the same learner characteristics and intervention evidence.
- A **scenario** is one specific situation in which the student exhibits a challenging behavior, captured as a 6-turn tutor–student dialogue. Each scenario belongs to exactly one category.
- A **rubric** is a checklist of evidence-suppotrted response principles used to evaluate open-ended responses, defined per category and **response type** (immediate or long-term).

Each scenario may generate up to two **evaluation prompts** — an *immediate* prompt asking what the tutor should say or do right now, and a *long-term* prompt asking what sustained interventions to implement over several weeks. A prompt is created only when the scenario includes a matching effective strategy in the source paper.

Rubrics are defined at the **category level**, not the scenario level. A single immediate rubric and a single long-term rubric cover all scenarios belonging to the same student category. Each rubric criterion represents an independent, evidence-supported response principle — satisfying any one criterion constitutes a successful response.

---

## ✍️ Structural Relationships

```
┌─────────────────────────────────────────────────────────┐
│                        CATEGORY                         │
│  category_id · category_title · student_description     │
└───────────────────────┬─────────────────────────────────┘
                        │  1 category : N scenarios
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
┌──────────────────┐        ┌────────────────────────────────────────┐
│      RUBRIC      │        │               SCENARIO                 │
│  (per category)  │        │  scenario_id · minimal_context         │
│                  │        │  dialogue · (full_context)             │
│  ┌────────────┐  │        └───────────────────┬────────────────────┘
│  │ immediate  │  │                            │  1 scenario : 1–2 prompts
│  │  rubric    │  │          ┌─────────────────┴──────────────────┐
│  └────────────┘  │          │                                    │
│  ┌────────────┐  │          ▼                                    ▼
│  │ long-term  │  │  ┌──────────────────┐             ┌──────────────────┐
│  │  rubric    │  │  │ DIALOGUE PROMPT  │             │ DIALOGUE PROMPT  │
│  └────────────┘  │  │   (immediate)    │             │   (long-term)    │
└──────────────────┘  │                  │             │                  │
          │           │  dialogue_prompt │             │  dialogue_prompt │
          │           │  _id · type      │             │  _id · type      │
          │           └────────┬─────────┘             └────────┬─────────┘
          │                    │                                │
          │      ┌─────────────┘                    ┌───────────┘
          │      │                                  │
          │      ▼                                  ▼
          │  ┌───────────────────────┐   ┌───────────────────────┐
          └─►│  RUBRIC EVAL PROMPT   │   │  RUBRIC EVAL PROMPT   │
             │     (immediate)       │   │     (long-term)       │
             └───────────────────────┘   └───────────────────────┘
           
                      1 scenario : N MCQs
                               │
                     ┌─────────┴──────────┐
                     ▼                    ▼
              ┌─────────────┐    ┌─────────────┐
              │   MCQ (im)  │    │   MCQ (lt)  │
              │ mcq_id      │    │ mcq_id      │
              │ options     │    │ options     │
              │ correct_ans │    │ correct_ans │
              └─────────────┘    └─────────────┘

 Rubric ──── applied to ────► Rubric Eval Prompt
 (category-level)             (dialogue-prompt-level)
```

---

## 📚 Dataset Files

### `dataset_overall.csv`

The central index table linking all benchmark components. Each row corresponds to one **dialogue prompt** (one open-ended evaluation item).

| Column | Description |
|---|---|
| `dialogue_prompt_id` | 4-digit unique ID for this evaluation prompt (e.g. `0001`) |
| `category_id` | 3-digit category ID (e.g. `001`) |
| `category_title` | Human-readable student category name |
| `scenario_id` | 3-digit scenario ID (e.g. `001`); shared across immediate and long-term prompts of the same case |
| `scenario_title` | Title of the specific scenario |
| `response_type` | `immediate` or `long_term` |
| `minimal_context` | One-sentence task-and-setting description shown to the responder |
| `dialogue` | Full tutor–student dialogue text (TUTOR:/STUDENT: format) |
| `mcq_ids` | Pipe-separated list of MCQ IDs linked to this scneario/dialogue prompt (e.g. `0001\|0002`) |
| `challenge_type` | Type of challenging behavior exhibited (e.g. `Emotional Distress`) |
| `rubric_criteria_immediate` | Formatted immediate rubric criteria for this category |
| `rubric_criteria_long_term` | Formatted long-term rubric criteria for this category |

### `document_list.csv`

A registry of the source research papers from which student cases were drawn. Each row corresponds to one paper. The list contains papers that passed the pipeline's dialogue synthesis eligibility filter — i.e., papers judged to contain sufficient student observation data and teacher guidance to support evidence-grounded dialogue generation. The `in_dataset` column marks with `"O"` the papers whose cases were ultimately included in the benchmark dataset.

| Column | Description |
|---|---|
| `title` | Title of the source paper |
| `authors` | Author(s) of the paper |
| `year` | Publication year |
| `eric_id` | ERIC database identifier (e.g. `EJ1194362`) |
| `eric_page` | Direct URL to the paper's ERIC page |
| `abstract` | Abstract of the paper as retrieved from ERIC |
| `in_dataset` | `"O"` if the paper's cases are included in the benchmark; empty otherwise |

---

## 📂 Folder Structure

```
benchmark/
├── dataset_overall.csv                # Central index (see above)
├── document_list.csv                  # Source paper registry
│
├── dialogue_prompts/                  # Open-ended evaluation prompt TXTs
│   └── dialogue_prompt_{id}_{type}_c{cat}_s{scen}.txt
│
├── dialogue/                          # Scenario JSONs (compact)
│   └── scenario_{scen}_c{cat}_{suffix}.json
│
├── dialogue_full_context/             # Scenario JSONs (with full context)
│   └── scenario_{scen}_c{cat}_{suffix}.json
│
├── rubric/                            # Rubric JSONs (per category × type)
│   └── rubric_c{cat}_{type}.json
│
├── rubric_evaluation_prompts/         # Filled rubric scoring prompts (TUTOR_RESPONSE placeholder only)
│   └── rubric_evaluation_prompt_{id}_{type}_c{cat}_s{scen}.txt
│
├── mcq/                               # MCQ data JSONs
│   └── mcq_{id}_{type}_c{cat}_s{scen}.json
│
└── mcq_prompt/                        # MCQ evaluation prompt TXTs
    └── mcq_{id}_{type}_c{cat}_s{scen}.txt
```

**Filename suffixes for scenario files:**

| Suffix | Meaning |
|---|---|
| `_im` | Only an immediate dialogue prompt exists for this scenario |
| `_lt` | Only a long-term dialogue prompt exists for this scenario |
| `_bt` | Both immediate and long-term prompts exist |

---

## 📄 File Descriptions

### `dialogue_prompts/`

Plain-text files containing the open-ended evaluation prompt shown to a model. Each file contains the scenario context (minimal context), the tutor–student dialogue, and the question prompt for the specified response type.

**Naming:** `dialogue_prompt_{id}_{type}_c{cat}_s{scen}.txt`

---

### `dialogue/`

Compact scenario JSON files. Contains the core information needed to understand a scenario without the full annotation context.

**Fields:**

| Field | Description |
|---|---|
| `category_id` | 3-digit category ID |
| `category_title` | Student category name |
| `scenario_id` | 3-digit scenario ID |
| `scenario_title` | Scenario title |
| `minimal_context` | One-sentence task-and-setting summary |
| `dialogue` | List of turn objects `{turn_id, speaker, message, ...}` |
| `dialogue_combined` | Same dialogue as a single `TUTOR: … \n\n STUDENT: …` string |
| `immediate_dialogue_prompt` | Linked immediate `dialogue_prompt_id` (int or null) |
| `long_term_dialogue_prompt` | Linked long-term `dialogue_prompt_id` (int or null) |
| `challenge_type` | Type of challenging behavior |

**Naming:** `scenario_{scen}_c{cat}_{suffix}.json`

---

### `dialogue_full_context/`

Extended version of the scenario JSON, adding a `full_context` block. Used for rubric-based scoring where the grader requires full annotation detail.

**Additional field:**

| Field | Description |
|---|---|
| `full_context` | Structured object with `content → {task, context, behavior_patterns, challenging_behavior_trigger}` |

**Naming:** `scenario_{scen}_c{cat}_{suffix}.json`

---

### `rubric/`

Rubric JSON files, one per category × response type. Each rubric is a checklist of independent, evidence-supported response principles for a specific student profile.

**Fields:**

| Field | Description |
|---|---|
| `category_id` | 3-digit category ID |
| `category_title` | Student category name |
| `rubric_type` | `immediate` or `long_term` |
| `student_description` | Short prose summary of the student's needs and characteristics |
| `rubric` | Object containing `criteria` — a list of `{criterion_id, criterion_name, rationale, check}` |
| `challenge_type` | Primary challenge type for this category |

Criteria are scored independently (0 or 1). A response earns credit for satisfying any one criterion — no single response is expected to cover all principles.

**Naming:** `rubric_c{cat}_{type}.json`

---

### `rubric_evaluation_prompts/`

Pre-filled scoring prompt TXTs. All fields except `{TUTOR_RESPONSE}` are populated: the rubric criteria, student description, dialogue, and full context are inserted. A model response can be dropped in directly for automated scoring.

**Naming:** `rubric_evaluation_prompt_{id}_{type}_c{cat}_s{scen}.txt`

---

### `mcq/`

MCQ data JSON files. Each file represents one 4-option multiple-choice question derived from a scenario. The correct option is one effective strategy from the target scenario; the three distractors are effective strategies drawn from other scenarios that share a similar classroom context but address a different learner problem.

**Fields:**

| Field | Description |
|---|---|
| `mcq_id` | 4-digit MCQ ID |
| `category_id` | Category this MCQ belongs to |
| `category_title` | Student category name |
| `scenario_id` | Source scenario ID |
| `dialogue_prompt_id` | Linked dialogue prompt ID |
| `mcq_type` | `immediate` or `long_term` |
| `minimal_context` | One-sentence scenario context |
| `dialogue` | Clean TUTOR:/STUDENT: dialogue text |
| `options` | Dict of `{A, B, C, D}` option texts |
| `correct_answer` | Letter of the correct option |
| `challenge_type` | Challenging behavior type |

**Naming:** `mcq_{id}_{type}_c{cat}_s{scen}.json`

---

### `mcq_prompt/`

Plain-text MCQ evaluation prompts. Each file contains the scenario context, dialogue, and answer options, and ends with a JSON response format specification.

**Naming:** `mcq_{id}_{type}_c{cat}_s{scen}.txt`

---

## 🔖 ID System

All IDs are assigned sequentially and zero-padded.

| ID | Padding | Scope |
|---|---|---|
| `category_id` | 3 digits (`001`, `002`, …) | Unique per student category |
| `scenario_id` | 3 digits | Unique globally across all scenarios |
| `dialogue_prompt_id` | 4 digits (`0001`, `0002`, …) | Unique per evaluation prompt |
| `mcq_id` | 4 digits | Unique per MCQ |

> **Note on `scenario_id`:** Immediate and long-term dialogue prompts derived from the same case share the same `scenario_id`. The `dialogue_prompt_id` is what uniquely identifies each individual prompt.

---

## 🎒 Challenge Types

Each student category is assigned one of five challenge types, representing the primary observable behavior a tutor must respond to. The taxonomy was designed and refined by an author with a doctoral degree in education.

| Challenge Type | Description |
|---|---|
| `Task Resistance` | Student actively refuses or avoids engaging with an assigned task |
| `Emotional Distress` | Student exhibits emotional dysregulation such as crying, frustration, or anxiety |
| `Attention/Engagement Failure` | Student is off-task, distracted, or disengaged without overt resistance |
| `Social-Interpersonal Difficulty` | Student struggles with peer or adult interactions in the learning context |
| `Academic Struggle` | Student has difficulty with the academic content itself |

---

## 📊 Benchmark Statistics

| Unit | Count |
|---|---|
| Source papers | 410 |
| Papers used in dataset (filtered for dialogue synthesis) | 209 |
| Student categories | 335 |
| Scenarios | 621 |
| Dialogue prompts — immediate | 567 |
| Dialogue prompts — long-term | 513 |
| Dialogue prompts — total | 1,080 |
| Rubrics — immediate | 321 |
| Rubrics — long-term | 322 |
| MCQs | 1,727 |

---

## 🛠️ Pipeline Code

The data construction pipeline used to build this benchmark is located in the `benchmark_code/` folder.

### Key Files

| File | Description |
|------|-------------|
| `main.py` | Runs Pipelines 1–5 sequentially on a single PDF — screen, analyze, extract, build scenario, synthesize dialogue, and collect model responses |
| `evaluation_pipeline.py` | Generates per-category rubrics and evaluates collected model responses using a 0/1 checklist rubric |
| `quaternary_pipeline.py` | Builds 4-option MCQ questions via embedding-based distractor selection, runs models, and summarizes accuracy |
| `run_evaluation.py` | Batch runner that auto-discovers all response files and runs evaluation_pipeline.py across all categories |
| `run_study_group.py` | Top-level orchestrator: runs the full pipeline on a PDF directory, then proceeds to rubric generation and evaluation |

### Directory Structure

```
benchmark_code/
├── main.py
├── evaluation_pipeline.py
├── quaternary_pipeline.py
├── run_evaluation.py
├── run_study_group.py
│
├── config/
│   ├── config_default_model.json    # Default LLM per pipeline stage
│   ├── config_filter.json           # Eligibility thresholds and scenario count
│   ├── config_dialogue_turn.json    # Dialogue turn count range
│   ├── config_response.json         # Models for response collection
│   └── config_evaluation.json       # Models for rubric generation and evaluation
│
├── prompts/                         # {KEY}-style prompt templates for each pipeline stage
└── utils/                           # PDF extraction, LLM calls, prompt building, file I/O
```

For full usage instructions, see `benchmark_code/README.md`.