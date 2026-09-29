# DiaBench

Code and data for:

> **DiaBench: A Benchmarking Framework to Evaluate the Performance of LLMs in
> T2D Education**
> Yang Li, Syna, Chuyi Zhang, Bhavana Kunisetty, Elizabeth Healey, Agatha F.
> Scheideman, Han Meng, Mandy M. Shao, Anna Simos, Helge Ræder, Yu Kuei Lin,
> Jacob Kohlenberg, Yangfu Zhang, David C. Klonoff, Marina Basina, Michael
> Snyder, Haipeng Chen, Tao Wang.

DiaBench is a standardized, MCQ-based benchmark for evaluating LLM accuracy
and self-consistency on type 2 diabetes (T2D) patient education, built from
96 clinician-curated Q&A pairs and three literacy-level configurations. See
`CITATION.cff` for the full author list, and cite the paper if you use this
benchmark.

## Pipeline overview

This mirrors Figure 1 of the manuscript:

```
 STEP 1                STEP 2                   STEP 3                  STEP 4
 CDCESs, MDs, and   →  Generate MCQs         →  7 LLMs perform       →  Assess
 endocrinologists      using GPT-5/Gemini        evaluation              performance
 curate Q&A pairs      (3 configurations)        (zero-shot,
                                                   3 runs each)

 original_questions.csv  generate_mcq.py        evaluation/
                          Prompt_template/
                          Generated_MCQs/
```

- **Configuration 1** — 9th-grade literacy, 4 options, 1 correct answer
- **Configuration 2** — undergraduate literacy, 7 options, 1 correct answer
- **Configuration 3** — graduate literacy, 10 options, 2 correct answers

Internally the evaluation scripts call these `easy` / `medium` / `hard`
respectively (auto-detected from how many lettered options are present) —
see `evaluation/common.py`.

## Repo layout

```
DiaBench/
├── original_questions.csv        the 96 source Q&A pairs (Step 1)
├── Prompt_template/
│   └── mcq_generation_template.py    the 3 configuration prompts (Appendix B)
├── generate_mcq.py                MCQ generation via Gemini or GPT-5 (Step 2)
├── Generated_MCQs/
│   ├── MCQs_Config1.csv           96 generated MCQs, Configuration 1
│   ├── MCQs_Config2.csv           96 generated MCQs, Configuration 2
│   └── MCQs_Config3.csv           96 generated MCQs, Configuration 3
├── evaluation/                    Step 3: run each of the 7 LLMs zero-shot,
│   │                              3x for self-consistency
│   ├── common.py                  shared config/parsing helpers
│   ├── cohere_evaluation.py
│   ├── gemini_evaluation.py
│   ├── openai_evaluation.py           (GPT-5)
│   ├── llama4_evaluation.py           (Together AI batch API)
│   ├── deepseek_evaluation.py         (Together AI batch API, DeepSeek-R1)
│   ├── mistral_evaluation.py
│   └── perplexity_evaluation.py
├── requirements.txt
├── .env.example
├── LICENSE
└── CITATION.cff
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

cp .env.example .env            # then fill in your own API keys — see .env.example
```

## Generating MCQs

```bash
python generate_mcq.py --llm gemini --prompt_template CONFIG1 \
    --input-file original_questions.csv --output-dir Generated_MCQs

# --llm gpt and --prompt_template CONFIG2 / CONFIG3 work the same way
```

The `Generated_MCQs/` folder already has the 96-question output for all three
configurations checked in, so this step only needs to be rerun if the source
Q&A pairs or prompts change.

## Running an evaluation

Each script in `evaluation/` follows the same interface:

```bash
cd evaluation

# Explicit difficulty (defaults to the matching Generated_MCQs/ file)
python cohere_evaluation.py     --difficulty easy
python gemini_evaluation.py     --difficulty medium
python openai_evaluation.py     --difficulty hard
python mistral_evaluation.py    --difficulty easy
python perplexity_evaluation.py --difficulty medium

# Together AI batch-mode scripts additionally support --limit for a quick test run
python llama4_evaluation.py     --difficulty hard --limit 5
python deepseek_evaluation.py   --difficulty hard --limit 5

# Or point at any file directly
python cohere_evaluation.py --file ../Generated_MCQs/MCQs_Config1.csv
```

If `--difficulty` is omitted, it's auto-detected from the option count in the
loaded data. Every script implements the same protocol: zero-shot prompting,
3 independent runs per question, majority vote for the final answer, and a
self-consistency flag (do all 3 runs agree). Results are written locally to
`results/config{1,2,3}_<difficulty>/` (gitignored — these are local run
outputs, not checked into the repo).

## Data sharing

Per the manuscript's Data Sharing Statement, code and data are open-sourced
here: `original_questions.csv` (the 96 source Q&A pairs) and
`Generated_MCQs/` (the resulting MCQs for all three configurations) are the
actual data behind the paper.
