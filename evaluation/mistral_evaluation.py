import os
import time
import re
import datetime
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
import argparse

from common import (
    QUESTION_CONFIG,
    RETRY_BACKOFF_SECONDS,
    PER_RUN_DELAY_SECONDS,
    PER_QUESTION_DELAY_SECONDS,
    get_letters_for_difficulty,
    detect_difficulty_from_options,
    extract_answer_letters,
    save_df_excel_or_csv,
)

print("=== MISTRAL AI EVALUATION PIPELINE ===")

# -------------------------
# Helpers (provider-specific — parsing/prompting for Mistral's response style)
# -------------------------


def create_eval_prompt(question: str, options: str, allowed_letters: List[str], num_correct: int = 1) -> str:
    if num_correct == 1:
        return f"""You are a medical expert evaluating a multiple-choice question. Please analyze the question and options carefully, then provide the correct answer.

Question: {question}

Options:
{options}

Allowed answer letters: {', '.join(allowed_letters)}

Please think through this step by step:
1. Read the question carefully
2. Consider each option
3. Identify the correct answer
4. Provide your answer in the format: "The correct answer is [LETTER]."

The correct answer is:"""
    else:
        return f"""You are a medical expert evaluating a multiple-choice question with multiple correct answers. Please analyze the question and options carefully, then provide the correct answers.

Question: {question}

Options:
{options}

Allowed answer letters: {', '.join(allowed_letters)}
Number of correct answers: {num_correct}

Please think through this step by step:
1. Read the question carefully
2. Consider each option
3. Identify the {num_correct} correct answer(s)
4. Provide your answer in the format: "The correct answers are [LETTER1, LETTER2]."

The correct answers are:"""


def build_mistral_client():
    """Build Mistral client (SDK >= 1.9)."""
    load_dotenv()
    try:
        from mistralai import Mistral
        # message classes come from mistralai.models in 1.9.x
        from mistralai.models import UserMessage  # noqa: F401  (import check)
    except Exception as e:
        raise RuntimeError("mistralai >= 1.9 not installed. Try: pip install -U mistralai") from e

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY in environment/.env")

    client = Mistral(api_key=api_key)
    model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    return client, model_name


def get_mistral_response(client, model_name: str, prompt: str, allowed_letters: List[str],
                         num_runs: int = 3, num_correct: int = 1):
    from mistralai.models import UserMessage

    all_answers: List[List[str]] = []
    all_responses: List[str] = []

    for run in range(num_runs):
        last_err = None
        # retry with backoff
        for backoff in [0.0] + RETRY_BACKOFF_SECONDS:
            if backoff:
                time.sleep(backoff)
            try:
                resp = client.chat.complete(
                    model=model_name,
                    messages=[UserMessage(content=prompt)],
                    temperature=0.0,
                    max_tokens=1000,
                )
                # --- extract text (list of chunks with .text) ---
                parts = resp.choices[0].message.content
                if isinstance(parts, list):
                    response_text = "".join(getattr(p, "text", "") for p in parts)
                else:
                    response_text = str(parts)
                # fallback some SDKs expose aggregated field
                if not response_text:
                    response_text = getattr(resp, "output_text", "") or ""
                # ------------------------------------------------
                all_responses.append(response_text)

                # parse picks (your existing parser is fine)
                txt_up = response_text.upper()

                if num_correct == 1:
                    patts = [
                        r'\bTHE\s+CORRECT\s+ANSWER\s+IS\s*[:\-]?\s*([A-J])\b',
                        r'\bANSWER\s*[:\-]?\s*([A-J])\b',
                        r'([A-J])\)',
                    ]
                    pick = None
                    for pat in patts:
                        m = re.search(pat, txt_up)
                        if m and m.group(1) in allowed_letters:
                            pick = m.group(1)
                            break
                    if not pick:
                        for t in re.findall(r'\b([A-J])\b', txt_up):
                            if t in allowed_letters:
                                pick = t
                                break
                    all_answers.append([pick] if pick else [])
                else:
                    pick = []
                    for pat in [
                        r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*\[([A-J](?:\s*,\s*[A-J])*)\]',
                        r'\[([A-J](?:\s*,\s*[A-J])*)\]',
                        r'([A-J](?:\s*,\s*[A-J])*)',
                    ]:
                        m = re.search(pat, response_text, flags=re.IGNORECASE)
                        if m:
                            letters = re.findall(r'[A-J]', m.group(1).upper())
                            pick = [l for l in letters if l in allowed_letters][:num_correct]
                            if len(pick) == num_correct:
                                break
                    all_answers.append(pick if pick else [])
                break  # success; exit retry loop
            except Exception as e:
                last_err = e
                continue

        if last_err and len(all_responses) <= run:
            all_responses.append(f"ERROR: {last_err}")
            all_answers.append([])

        if run < num_runs - 1:
            time.sleep(PER_RUN_DELAY_SECONDS)

    # consistency + majority
    if num_correct == 1:
        non_empty = [a for a in all_answers if a]
        is_consistent = len(set(tuple(x) for x in non_empty)) == 1 if non_empty else False
        flat = [a[0] for a in non_empty]
        final = [Counter(flat).most_common(1)[0][0]] if flat else []
    else:
        sets = [tuple(sorted(a)) for a in all_answers if a]
        is_consistent = len(set(sets)) == 1 if sets else False
        final = list(Counter(sets).most_common(1)[0][0]) if sets else []

    combined_text = (
        "SELF-CONSISTENCY RESULTS\n"
        f"Runs: {num_runs}\n"
        f"Answers: {all_answers}\n"
        f"Consistent: {is_consistent}\n\n"
        "=== RESPONSE #1 ===\n"
        f"{all_responses[0] if all_responses else ''}"
    )
    return final, combined_text, all_answers, is_consistent


def process_question(client, model_name: str, row: pd.Series, num_runs: int = 3) -> Dict:
    """Process a single question with Mistral AI."""
    question_id = row.get('ID', None)
    question = row.get('Generated Question', '')
    options = row.get('Choices', '')
    correct_answer = row.get('Answer', '')
    
    # Auto-detect difficulty
    diff = detect_difficulty_from_options(options)
    allowed = get_letters_for_difficulty(diff)
    num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]
    
    prompt = create_eval_prompt(question, options, allowed, num_correct)

    print(f"Processing question {question_id if question_id is not None else ''} (difficulty: {diff}, num_correct: {num_correct})")
    picks, full_text, all_answers, is_consistent = get_mistral_response(
        client, model_name, prompt, allowed, num_runs=num_runs, num_correct=num_correct
    )

    gold = extract_answer_letters(correct_answer, allowed, num_correct)
    match = (picks == gold) if num_correct == 1 else (set(picks) == set(gold))

    return {
        'ID': question_id,
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Correct_Answers_Parsed': ','.join(gold),
        'Mistral_Answer': ','.join(picks),
        'Full_Response': full_text,
        'Match': match,
        'Self_Consistent': 'Yes' if is_consistent else 'No',
        'Difficulty': diff,
        'Num_Correct': num_correct,
        'All_Answers': str(all_answers),
        'Is_Consistent': is_consistent
    }


DIFFICULTY_TO_FOLDER = {
    "easy": "config1_easy",
    "medium": "config2_medium",
    "hard": "config3_hard",
}

DIFFICULTY_TO_DATA_FILE = {
    "easy": os.path.join("..", "Generated_MCQs", "MCQs_Config1.csv"),
    "medium": os.path.join("..", "Generated_MCQs", "MCQs_Config2.csv"),
    "hard": os.path.join("..", "Generated_MCQs", "MCQs_Config3.csv"),
}

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Mistral AI MCQ evaluation")
    parser.add_argument("--file", dest="data_file", default=None,
                         help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv.")
    parser.add_argument("--difficulty", dest="difficulty", default=None,
                         choices=["easy", "medium", "hard"],
                         help="easy=Config 1, medium=Config 2, hard=Config 3. "
                              "Auto-detected from --file if omitted.")
    args = parser.parse_args()

    # Setup
    client, model_name = build_mistral_client()
    print(f"1. Mistral AI model: {model_name}")

    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE.get(args.difficulty, DIFFICULTY_TO_DATA_FILE["easy"])
    df = pd.read_csv(data_file) if data_file.lower().endswith('.csv') else pd.read_excel(data_file)
    print(f"2. Data loaded: {len(df)} questions from {data_file}")

    difficulty = args.difficulty or detect_difficulty_from_options(
        df.iloc[0]["Choices"] if not df.empty else ""
    )
    out_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[difficulty])
    os.makedirs(out_dir, exist_ok=True)
    print(f"3. Results dir: {out_dir} (difficulty: {difficulty})")

    checkpoint_path = os.path.join(out_dir, "mistral_results_checkpoint.csv")

    num_runs = 3
    print(f"4. Processing with {num_runs} runs per question (free-tier delays)")

    results = []
    for i, row in df.iterrows():
        print(f"\n--- Question {i+1}/{len(df)} ---")
        res = process_question(client, model_name, row, num_runs)
        results.append(res)
        # checkpoint append
        try:
            pd.DataFrame([res]).to_csv(
                checkpoint_path,
                mode="a",
                header=not os.path.exists(checkpoint_path),
                index=False,
                encoding="utf-8",
            )
        except Exception:
            pass
        time.sleep(PER_QUESTION_DELAY_SECONDS)

    results_df = pd.DataFrame(results)
    acc = float(results_df['Match'].mean() * 100) if not results_df.empty else 0.0
    self_cons = float(results_df['Is_Consistent'].mean() * 100) if not results_df.empty else 0.0

    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Self-consistency: {self_cons:.2f}%")

    out_main = os.path.join(out_dir, f'mistral_results_{difficulty}_with_consistency.xlsx')
    save_df_excel_or_csv(results_df, out_main)

    summary = pd.DataFrame([{
        'difficulty': difficulty,
        'total_questions': len(df),
        'accuracy': acc,
        'self_consistency': self_cons
    }])
    save_df_excel_or_csv(summary, os.path.join(out_dir, f'mistral_summary_{difficulty}.xlsx'))

    print("=== MISTRAL AI EVALUATION COMPLETE ===")
