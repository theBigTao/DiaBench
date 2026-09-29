import os
import time
import re
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
import argparse
import requests

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

print("=== PERPLEXITY EVALUATION PIPELINE ===")

# -------------------------
# Helpers (provider-specific — parsing/prompting for Perplexity's response style)
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


# -------------------------
# OpenRouter client (for Perplexity models)
# -------------------------
def build_perplexity_client():
    try:
        load_dotenv()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
    # Read from environment / .env — never hardcode credentials here.
    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Perplexity API key")
    
    # Use Perplexity API directly
    model_name = os.getenv("PPLX_MODEL", "sonar")
    endpoint = "https://api.perplexity.ai/chat/completions"
    return api_key, model_name, endpoint


def call_perplexity_with_retry(api_key: str, model_name: str, endpoint: str, prompt: str) -> str:
    last_err = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    for delay in [0.0] + RETRY_BACKOFF_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    return content or ""
                except Exception:
                    # Fallback shapes
                    return str(data)
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = e
    print(f"  Perplexity API call failed after retries: {last_err}")
    return ""


def get_perplexity_response(api_key: str, model_name: str, endpoint: str, prompt: str,
                            allowed_letters: List[str], num_runs: int = 3, num_correct: int = 1) -> Tuple[List[str], str, List[List[str]], bool]:
    all_answers: List[List[str]] = []
    all_responses: List[str] = []

    for run in range(num_runs):
        full_text = call_perplexity_with_retry(api_key, model_name, endpoint, prompt)
        all_responses.append(full_text)

        txt_up = full_text.upper()
        if num_correct == 1:
            patts = [
                r'\bTHE\s+CORRECT\s+ANSWER\s+IS\s*[:\-]?\s*\*?\*?([A-J])\*?\*?\b',
                r'\bANSWER\s*[:\-]?\s*\*?\*?([A-J])\*?\*?\b',
                r'([A-J])\)',
                r'Therefore, the correct answer is\s*\*?\*?([A-J])\*?\*?',
                r'Therefore, option\s*\*?\*?([A-J])\*?\*?',
                r'option\s*\*?\*?([A-J])\*?\*?\s*best describes',
                r'option\s*\*?\*?([A-J])\*?\*?\s*is correct',
                r'The correct answer is\s*\*?\*?([A-J])\*?\*?',
                r'Answer:\s*\*?\*?([A-J])\*?\*?',
                r'\[([A-J])\]',
                r'\(([A-J])\)'
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
            pick: List[str] = []
            for pat in [
                r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*\[([A-J](?:\s*,\s*[A-J])*)\]',
                r'\[([A-J](?:\s*,\s*[A-J])*)\]',
                r'([A-J](?:\s*,\s*[A-J])*)',
                r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*\*?\*?([A-J])\*?\*?\s*AND\s*\*?\*?([A-J])\*?\*?',
                r'ANSWERS\s*[:\-]?\s*\*?\*?([A-J])\*?\*?\s*AND\s*\*?\*?([A-J])\*?\*?',
                r'([A-J])\s*AND\s*([A-J])',
                r'([A-J])\s*,\s*([A-J])'
            ]:
                m = re.search(pat, full_text, flags=re.IGNORECASE)
                if m:
                    if len(m.groups()) == 2:  # For patterns with two capture groups (A AND B)
                        letters = [m.group(1), m.group(2)]
                    else:  # For patterns with one capture group
                        letters = re.findall(r'[A-J]', m.group(1).upper())
                    pick = [l for l in letters if l in allowed_letters][:num_correct]
                    if len(pick) == num_correct:
                        break
            all_answers.append(pick if pick else [])

        if run < num_runs - 1:
            time.sleep(PER_RUN_DELAY_SECONDS)

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
    )
    
    # Add all responses
    for i, response in enumerate(all_responses):
        combined_text += f"=== RESPONSE #{i+1} ===\n{response}\n\n"
    return final, combined_text, all_answers, is_consistent


def process_question(api_key: str, model_name: str, endpoint: str, row: pd.Series, num_runs: int = 3) -> Dict:
    question_id = row.get('ID', None)
    question = row.get('Generated Question', '')
    options = row.get('Choices', '')
    correct_answer = row.get('Answer', '')

    diff = detect_difficulty_from_options(options)
    allowed = get_letters_for_difficulty(diff)
    num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]

    prompt = create_eval_prompt(question, options, allowed, num_correct)

    print(f"Processing question {question_id if question_id is not None else ''} (difficulty: {diff}, num_correct: {num_correct})")
    picks, full_text, all_answers, is_consistent = get_perplexity_response(
        api_key, model_name, endpoint, prompt, allowed, num_runs=num_runs, num_correct=num_correct
    )

    gold = extract_answer_letters(correct_answer, allowed, num_correct)
    match = (picks == gold) if num_correct == 1 else (set(picks) == set(gold))

    return {
        'ID': question_id,
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Correct_Answers_Parsed': ','.join(gold),
        'Perplexity_Answer': ','.join(picks),
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
    parser = argparse.ArgumentParser(description="Perplexity Sonar MCQ evaluation")
    parser.add_argument("--file", dest="data_file", default=None,
                         help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv.")
    parser.add_argument("--difficulty", dest="difficulty", default=None,
                         choices=["easy", "medium", "hard"],
                         help="easy=Config 1, medium=Config 2, hard=Config 3. "
                              "Auto-detected from --file if omitted.")
    args = parser.parse_args()

    # Setup
    api_key, model_name, endpoint = build_perplexity_client()
    print(f"1. Perplexity model: {model_name}")

    # Read data (generator output)
    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE.get(args.difficulty, DIFFICULTY_TO_DATA_FILE["easy"])
    df = pd.read_csv(data_file) if data_file.lower().endswith('.csv') else pd.read_excel(data_file)
    print(f"2. Data loaded: {len(df)} questions from {data_file}")

    num_runs = 3
    print(f"4. Processing with {num_runs} runs per question (free-tier delays)")

    # Determine difficulty and set checkpoint path
    difficulty = args.difficulty or detect_difficulty_from_options(df.iloc[0]['Choices'] if not df.empty else "")
    difficulty_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[difficulty])
    os.makedirs(difficulty_dir, exist_ok=True)
    print(f"3. Results dir: {difficulty_dir} (difficulty: {difficulty})")
    checkpoint_path = os.path.join(difficulty_dir, f'perplexity_results_checkpoint_{difficulty}.csv')
    
    results = []
    for i, row in df.iterrows():
        print(f"\n--- Question {i+1}/{len(df)} ---")
        res = process_question(api_key, model_name, endpoint, row, num_runs)
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

    out_main = os.path.join(difficulty_dir, f'perplexity_results_{difficulty}_with_consistency.xlsx')
    save_df_excel_or_csv(results_df, out_main)

    summary = pd.DataFrame([{
        'difficulty': difficulty,
        'total_questions': len(df),
        'accuracy': acc,
        'self_consistency': self_cons
    }])
    save_df_excel_or_csv(summary, os.path.join(difficulty_dir, f'perplexity_summary_{difficulty}.xlsx'))

    print("=== PERPLEXITY (VIA OPENROUTER) EVALUATION COMPLETE ===")


