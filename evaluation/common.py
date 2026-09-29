"""
Shared config and helpers used by every DiaBench single-model evaluator
(cohere_evaluation.py, gemini_evaluation.py, openai_evaluation.py,
llama4_evaluation.py, deepseek_evaluation.py, mistral_evaluation.py,
perplexity_evaluation.py).

These functions were byte-identical (or identical modulo quote style) across
all seven scripts in the original codebase, so they're consolidated here
rather than duplicated seven times. Extracting them does not change any
evaluator's behavior.

Difficulty naming: internally this uses "easy"/"medium"/"hard", which map to
the manuscript's Configuration 1/2/3 respectively:
  easy   == Configuration 1 (9th-grade literacy, 4 options, 1 correct answer)
  medium == Configuration 2 (undergraduate literacy, 7 options, 1 correct answer)
  hard   == Configuration 3 (graduate literacy, 10 options, 2 correct answers)
"""

from typing import Dict, List

QUESTION_CONFIG: Dict[str, Dict[str, int]] = {
    "easy": {"number_of_answer_options": 4, "number_of_correct_answer": 1},
    "medium": {"number_of_answer_options": 7, "number_of_correct_answer": 1},
    "hard": {"number_of_answer_options": 10, "number_of_correct_answer": 2},
}

# Shared retry/pacing defaults for the synchronous (non-batch) evaluators.
RETRY_BACKOFF_SECONDS = [2.0, 4.0, 8.0]
PER_RUN_DELAY_SECONDS = 6.0
PER_QUESTION_DELAY_SECONDS = 2.0


def get_letters_for_difficulty(difficulty: str) -> List[str]:
    """Return the list of valid answer letters (A, B, C, ...) for a difficulty."""
    if difficulty not in QUESTION_CONFIG:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    n_options = QUESTION_CONFIG[difficulty]["number_of_answer_options"]
    return [chr(ord("A") + i) for i in range(n_options)]


def detect_difficulty_from_options(options_text: str) -> str:
    """Infer easy/medium/hard from how many lettered options are present."""
    if not isinstance(options_text, str) or not options_text.strip():
        return "easy"
    lines = [ln.strip() for ln in options_text.split("\n") if ln.strip()]
    letters_found = []
    for ln in lines:
        if len(ln) >= 2 and ln[1] in [")", "."]:
            c = ln[0].upper()
            if "A" <= c <= "Z":
                letters_found.append(c)
    letters_found = list(dict.fromkeys(letters_found))
    if not letters_found or letters_found[0] != "A":
        return "easy"
    last = letters_found[-1]
    expected = [chr(ord("A") + i) for i in range(ord(last) - ord("A") + 1)]
    if letters_found != expected:
        return "easy"
    n = len(letters_found)
    if n == 4:
        return "easy"
    if n == 7:
        return "medium"
    if n == 10:
        return "hard"
    return "easy"


def extract_answer_letters(
    answer_text: str, allowed_letters: List[str], num_correct: int = 1
) -> List[str]:
    """Pull the gold-standard letter(s) out of the source dataset's Answer column."""
    import re

    import pandas as pd

    if pd.isna(answer_text):
        return []
    if num_correct == 1:
        if answer_text in allowed_letters:
            return [answer_text]
        m = re.search(r"Answer:\s*([A-Z])\)", answer_text)
        if m and m.group(1) in allowed_letters:
            return [m.group(1)]
        for l in allowed_letters:
            if l in answer_text:
                return [l]
        return []
    # multi (hard / Configuration 3)
    for pat in [
        r"Answer[s]?:\s*([A-Z](?:,\s*[A-Z])*)",
        r"\[([A-Z](?:,\s*[A-Z])*)\]",
        r"\(([A-Z](?:,\s*[A-Z])*)\)",
    ]:
        m = re.search(pat, answer_text)
        if m:
            picks = re.findall(r"[A-Z]", m.group(1))
            picks = [p for p in picks if p in allowed_letters][:num_correct]
            if len(picks) == num_correct:
                return picks
    found = []
    for l in allowed_letters:
        if l in answer_text:
            found.append(l)
            if len(found) == num_correct:
                break
    return found


def save_df_excel_or_csv(df, out_path: str) -> None:
    """Save a results DataFrame as .xlsx, falling back to .csv if that fails."""
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        df.to_excel(out_path, index=False)
        print(f"Saved Excel: {out_path}")
    except Exception as e:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"Excel failed ({e}); saved CSV: {csv_path}")
