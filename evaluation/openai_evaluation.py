import os
import time
import re
import argparse
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

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

print("=== OPENAI EVALUATION PIPELINE ===")

# -------------------------
# Helpers (provider-specific — parsing/prompting for GPT-5's response style)
# -------------------------


def create_eval_prompt(
    question: str, options: str, allowed_letters: List[str], num_correct: int
) -> str:
    allowed = ", ".join(allowed_letters)
    if num_correct == 1:
        return (
            f"Answer the question using one of the given choices.\n"
            f"{question}\n{options}\n\n"
            f'Begin your answer with this exact line (no extra text before it): "The correct answer is X"\n'
            f"Replace X with one of: {allowed}.\n"
            f"After that line, provide a short 1-2 sentence justification."
        )
    return (
        f"Answer the question using {num_correct} of the given choices.\n"
        f"{question}\n{options}\n\n"
        f'Begin your answer with this exact line (no extra text before it): "The correct answers are X, Y"\n'
        f"Replace X, Y with letters chosen from: {allowed}. Use a comma and a space between letters.\n"
        f"After that line, provide a short 1-2 sentence justification."
    )


def parse_model_answer(
    text: str, allowed_letters: List[str], num_correct: int
) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return ["ERROR"] if num_correct == 1 else ["ERROR"] * num_correct
    upper_text = text.upper()
    if num_correct == 1:
        m = re.search(r"THE\s+CORRECT\s+ANSWER\s+IS\s*[:\-]?\s*([A-J])\.?", upper_text)
        if m and m.group(1) in allowed_letters:
            return [m.group(1)]
    else:
        m = re.search(
            r"THE\s+CORRECT\s+ANSWERS?\s+ARE\s*[:\-]?\s*([A-J])\s*,\s*([A-J])\.?",
            upper_text,
        )
        if m:
            picks = [m.group(1), m.group(2)]
            picks = [p for p in picks if p in allowed_letters]
            if len(picks) == num_correct:
                return picks
    for pat in [
        r"\[([A-J](?:\s*,\s*[A-J]){0,9})\]",
        r"\(([A-J](?:\s*,\s*[A-J]){0,9})\)",
        r"\b([A-J](?:\s*,\s*[A-J]){1,9})\b" if num_correct > 1 else r"\b([A-J])\b",
    ]:
        m = re.search(pat, upper_text)
        if m:
            group = m.group(1)
            picks = [x.strip().upper() for x in group.split(",")]
            picks = [p for p in picks if p in allowed_letters]
            if num_correct == 1 and picks:
                return [picks[0]]
            if num_correct > 1 and len(picks) >= num_correct:
                return picks[:num_correct]
    tokens = re.findall(r"\b([A-J])\b", upper_text)
    tokens = [t for t in tokens if t in allowed_letters]
    if num_correct == 1:
        return [tokens[0]] if tokens else ["ERROR"]
    out = []
    for t in tokens:
        if t not in out:
            out.append(t)
        if len(out) == num_correct:
            break
    return out if out else ["ERROR"] * num_correct


# -------------------------
# OpenAI client (Chat API)
# -------------------------


def build_openai_client():
    load_dotenv()
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing openai SDK. pip install openai>=1.0.0") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    # The OpenAI() client reads OPENAI_API_KEY from environment
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    return client, model


def call_openai_with_retry(client, model: str, prompt: str) -> str:
    last_err = None
    for delay in [0.0] + RETRY_BACKOFF_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for multiple-choice evaluation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
            )
            text = resp.choices[0].message.content if resp.choices else ""
            return text or ""
        except Exception as e:
            last_err = e
    print(f"  OpenAI chat failed after retries: {last_err}")
    return ""


def get_openai_response(
    client,
    model: str,
    prompt: str,
    allowed_letters: List[str],
    num_runs: int,
    num_correct: int,
) -> Tuple[List[str], str, List[List[str]], bool]:
    all_answers: List[List[str]] = []
    full_responses: List[str] = []
    for run in range(num_runs):
        text = call_openai_with_retry(client, model, prompt)
        full_responses.append(text)
        picks = parse_model_answer(text, allowed_letters, num_correct)
        all_answers.append(picks)
        print(f"  Run {run+1}: Answer = {picks}")
        time.sleep(PER_RUN_DELAY_SECONDS)
    if num_correct == 1:
        consistent = all(a == all_answers[0] for a in all_answers)
    else:
        consistent = all(set(a) == set(all_answers[0]) for a in all_answers)
    combined = (
        f"SELF-CONSISTENCY RESULTS\nRuns: {num_runs}\nAnswers: {all_answers}\nConsistent: {consistent}\n"
        + ("\n\n=== RESPONSE #1 ===\n" + (full_responses[0] if full_responses else ""))
    )
    if num_correct == 1:
        flat = [a[0] for a in all_answers if a and a[0] != "ERROR"]
        final = [Counter(flat).most_common(1)[0][0]] if flat else ["ERROR"]
    else:
        tuples_ = [tuple(sorted(a)) for a in all_answers if a and "ERROR" not in a]
        final = (
            list(Counter(tuples_).most_common(1)[0][0])
            if tuples_
            else ["ERROR"] * num_correct
        )
    return final, combined, all_answers, consistent


# -------------------------
# Core pipeline
# -------------------------


def process_question(client, model: str, row: pd.Series, num_runs: int = 3) -> Dict:
    question = row["Generated Question"]
    question_id = row["ID"] if "ID" in row else None
    options = row["Choices"]
    correct_answer = row["Answer"]

    diff = detect_difficulty_from_options(options)
    allowed = get_letters_for_difficulty(diff)
    num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]

    prompt = create_eval_prompt(question, options, allowed, num_correct)

    print(
        f"Processing question {question_id if question_id is not None else ''} (difficulty: {diff}, num_correct: {num_correct})"
    )
    picks, full_text, all_answers, is_consistent = get_openai_response(
        client, model, prompt, allowed, num_runs=num_runs, num_correct=num_correct
    )

    gold = extract_answer_letters(correct_answer, allowed, num_correct)
    match = (picks == gold) if num_correct == 1 else (set(picks) == set(gold))

    return {
        "ID": question_id,
        "Question": question,
        "Options": options,
        "Correct_Answer": correct_answer,
        "Correct_Answers_Parsed": ",".join(gold),
        "OpenAI_Answer": ",".join(picks),
        "Full_Response": full_text,
        "Match": match,
        "Self_Consistent": "Yes" if is_consistent else "No",
        "Difficulty": diff,
        "Num_Correct": num_correct,
        "All_Answers": str(all_answers),
        "Is_Consistent": is_consistent,
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
    parser = argparse.ArgumentParser(description="OpenAI MCQ evaluation")
    parser.add_argument(
        "--file",
        dest="data_file",
        default=None,
        help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv.",
    )
    parser.add_argument(
        "--difficulty",
        dest="difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="easy=Config 1, medium=Config 2, hard=Config 3",
    )
    args = parser.parse_args()

    client, model_name = build_openai_client()
    print(f"1. OpenAI model: {model_name}")

    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE[args.difficulty]
    df = (
        pd.read_csv(data_file)
        if data_file.lower().endswith(".csv")
        else pd.read_excel(data_file)
    )
    print(f"2. Data loaded: {len(df)} questions from {data_file}")

    out_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[args.difficulty])
    CHECKPOINT_PATH = os.path.join(out_dir, "openai_results_checkpoint.csv")
    os.makedirs(out_dir, exist_ok=True)
    print(f"3. Results dir: {out_dir}")

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
                CHECKPOINT_PATH,
                mode="a",
                header=not os.path.exists(CHECKPOINT_PATH),
                index=False,
                encoding="utf-8",
            )
        except Exception:
            pass
        time.sleep(PER_QUESTION_DELAY_SECONDS)

    results_df = pd.DataFrame(results)
    acc = float(results_df["Match"].mean() * 100) if not results_df.empty else 0.0
    self_cons = (
        float(results_df["Is_Consistent"].mean() * 100) if not results_df.empty else 0.0
    )

    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Self-consistency: {self_cons:.2f}%")

    out_main = os.path.join(
        out_dir, f"openai_results_with_consistency_{args.difficulty}.xlsx"
    )
    save_df_excel_or_csv(results_df, out_main)

    summary = pd.DataFrame(
        [
            {
                "difficulty": f"{args.difficulty}",
                "total_questions": len(df),
                "accuracy": acc,
                "self_consistency": self_cons,
            }
        ]
    )
    save_df_excel_or_csv(
        summary, os.path.join(out_dir, f"openai_summary_{args.difficulty}.xlsx")
    )

    print("=== OPENAI EVALUATION COMPLETE ===")
