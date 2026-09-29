import os
import time
import re
import datetime
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

print("=== COHERE EVALUATION PIPELINE ===")

# -------------------------
# Helpers (provider-specific — parsing/prompting for Cohere's response style)
# -------------------------


def create_eval_prompt(question: str, options: str, allowed_letters: List[str], num_correct: int) -> str:
    allowed = ", ".join(allowed_letters)
    if num_correct == 1:
        return (
            f"Answer the question using one of the given choices.\n"
            f"{question}\n{options}\n\n"
            f"Begin your answer with this exact line (no extra text before it): \"The correct answer is X\"\n"
            f"Replace X with one of: {allowed}.\n"
            f"After that line, provide a short 1-2 sentence justification."
        )
    return (
        f"Answer the question using {num_correct} of the given choices.\n"
        f"{question}\n{options}\n\n"
        f"Begin your answer with this exact line (no extra text before it): \"The correct answers are X, Y\"\n"
        f"Replace X, Y with letters chosen from: {allowed}. Use a comma and a space between letters.\n"
        f"After that line, provide a short 1-2 sentence justification."
    )


def parse_model_answer(text: str, allowed_letters: List[str], num_correct: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return ["ERROR"] if num_correct == 1 else ["ERROR"] * num_correct
    upper_text = text.upper()
    if num_correct == 1:
        m = re.search(r'THE\s+CORRECT\s+ANSWER\s+IS\s*[:\-]?\s*([A-J])\.?', upper_text)
        if m and m.group(1) in allowed_letters:
            return [m.group(1)]
    else:
        m = re.search(r'THE\s+CORRECT\s+ANSWERS?\s+ARE\s*[:\-]?\s*([A-J])\s*,\s*([A-J])\.?', upper_text)
        if m:
            picks = [m.group(1), m.group(2)]
            picks = [p for p in picks if p in allowed_letters]
            if len(picks) == num_correct:
                return picks
    # bracket/comma
    for pat in [
        r'\[([A-J](?:\s*,\s*[A-J]){0,9})\]',
        r'\(([A-J](?:\s*,\s*[A-J]){0,9})\)',
        r'\b([A-J](?:\s*,\s*[A-J]){1,9})\b' if num_correct > 1 else r'\b([A-J])\b']:
        m = re.search(pat, upper_text)
        if m:
            group = m.group(1)
            picks = [x.strip().upper() for x in group.split(',')]
            picks = [p for p in picks if p in allowed_letters]
            if num_correct == 1 and picks:
                return [picks[0]]
            if num_correct > 1 and len(picks) >= num_correct:
                return picks[:num_correct]
    tokens = re.findall(r'\b([A-J])\b', upper_text)
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
# Cohere client (Chat API with fallback)
# -------------------------

def build_cohere_client():
    load_dotenv()
    try:
        import cohere
    except Exception as e:
        raise RuntimeError("Missing cohere SDK. pip install cohere") from e
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing COHERE_API_KEY in .env")
    client = cohere.ClientV2(api_key=api_key) if hasattr(cohere, 'ClientV2') else cohere.Client(api_key)
    # Default to a current chat model
    model = os.getenv("COHERE_MODEL", "command-a-03-2025")
    return client, model


def call_cohere_with_retry(client, model: str, prompt: str) -> str:
    last_err = None
    for delay in [0.0] + RETRY_BACKOFF_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            if hasattr(client, 'chat'):
                try:
                    resp = client.chat(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
                    if hasattr(resp, "message") and getattr(resp.message, "content", None):
                        parts = resp.message.content
                        texts = []
                        for p in parts:
                            t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                            if t:
                                texts.append(t)
                        if texts:
                            return "\n".join(texts)
                    text = getattr(resp, 'text', None)
                    if text:
                        return text
                except TypeError:
                    resp = client.chat(model=model, message=prompt, temperature=0.2)
                    text = getattr(resp, 'text', None) or getattr(resp, 'output_text', None)
                    if text:
                        return text
            if hasattr(client, 'generate'):
                gen = client.generate(model=model, prompt=prompt, max_tokens=512, temperature=0.2)
                text = None
                if hasattr(gen, 'generations') and gen.generations:
                    first = gen.generations[0]
                    text = getattr(first, 'text', None) or (first.get('text') if isinstance(first, dict) else None)
                if not text:
                    text = getattr(gen, 'text', None)
                return text or ""
            return ""
        except Exception as e:
            last_err = e
    print(f"  Cohere chat/generate failed after retries: {last_err}")
    return ""


def get_cohere_response(client, model: str, prompt: str, allowed_letters: List[str], num_runs: int, num_correct: int) -> Tuple[List[str], str, List[List[str]], bool]:
    all_answers: List[List[str]] = []
    full_responses: List[str] = []
    for run in range(num_runs):
        text = call_cohere_with_retry(client, model, prompt)
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
        final = list(Counter(tuples_).most_common(1)[0][0]) if tuples_ else ["ERROR"] * num_correct
    return final, combined, all_answers, consistent

# -------------------------
# Core pipeline
# -------------------------

def process_question(client, model: str, row: pd.Series, num_runs: int = 3) -> Dict:
    question = row['Generated Question']
    question_id = row['ID'] if 'ID' in row else None
    options = row['Choices']
    correct_answer = row['Answer']

    diff = detect_difficulty_from_options(options)
    allowed = get_letters_for_difficulty(diff)
    num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]

    prompt = create_eval_prompt(question, options, allowed, num_correct)

    print(f"Processing question {question_id if question_id is not None else ''} (difficulty: {diff}, num_correct: {num_correct})")
    picks, full_text, all_answers, is_consistent = get_cohere_response(
        client, model, prompt, allowed, num_runs=num_runs, num_correct=num_correct
    )

    gold = extract_answer_letters(correct_answer, allowed, num_correct)
    match = (picks == gold) if num_correct == 1 else (set(picks) == set(gold))

    return {
        'ID': question_id,
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Correct_Answers_Parsed': ','.join(gold),
        'Cohere_Answer': ','.join(picks),
        'Full_Response': full_text,
        'Match': match,
        'Self_Consistent': 'Yes' if is_consistent else 'No',
        'Difficulty': diff,
        'Num_Correct': num_correct,
        'All_Answers': str(all_answers),
        'Is_Consistent': is_consistent
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cohere MCQ evaluation")
    parser.add_argument(
        "--file", dest="data_file", default=None,
        help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv "
             "for each difficulty under --difficulty.",
    )
    parser.add_argument(
        "--difficulty", dest="difficulty", default=None,
        choices=["easy", "medium", "hard"],
        help="Which config to run (easy=Config 1, medium=Config 2, hard=Config 3). "
             "If omitted, it's auto-detected from the option count in --file.",
    )
    args = parser.parse_args()

    # Setup
    client, model_name = build_cohere_client()
    print(f"1. Cohere model: {model_name}")

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

    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE.get(args.difficulty, DIFFICULTY_TO_DATA_FILE["easy"])
    df = pd.read_csv(data_file)
    print(f"2. Data loaded: {len(df)} questions from {data_file}")

    difficulty = args.difficulty or detect_difficulty_from_options(
        df.iloc[0]["Choices"] if not df.empty else ""
    )
    out_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[difficulty])
    os.makedirs(out_dir, exist_ok=True)
    print(f"3. Results dir: {out_dir} (difficulty: {difficulty})")

    checkpoint_path = os.path.join(out_dir, "cohere_results_checkpoint.csv")

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

    out_main = os.path.join(out_dir, f'cohere_results_{difficulty}_with_consistency.xlsx')
    save_df_excel_or_csv(results_df, out_main)

    summary = pd.DataFrame([{
        'difficulty': difficulty,
        'total_questions': len(df),
        'accuracy': acc,
        'self_consistency': self_cons
    }])
    save_df_excel_or_csv(summary, os.path.join(out_dir, f'cohere_summary_{difficulty}.xlsx'))

    print("=== COHERE EVALUATION COMPLETE ===")
