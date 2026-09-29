import pandas as pd
import google.generativeai as genai
import os
import time
import re
from collections import Counter
from typing import Dict, List, Tuple
import datetime
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

print("=== GEMINI EVALUATION PIPELINE ===")


def parse_model_answer(text: str, allowed_letters: List[str], num_correct: int) -> List[str]:
    """Tolerant parser for model outputs: supports periods optional, brackets, commas, bare tokens."""
    if not isinstance(text, str) or not text.strip():
        return ["ERROR"] if num_correct == 1 else ["ERROR"] * num_correct

    # Normalize once
    upper_text = text.upper()

    # Preferred explicit line(s)
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

    # Bracket or generic comma forms
    patterns = [
        r'\[([A-J](?:\s*,\s*[A-J]){0,9})\]',
        r'\(([A-J](?:\s*,\s*[A-J]){0,9})\)',
        r'\b([A-J](?:\s*,\s*[A-J]){1,9})\b' if num_correct > 1 else r'\b([A-J])\b',
    ]
    for pat in patterns:
        m = re.search(pat, upper_text)
        if m:
            group = m.group(1)
            picks = [x.strip().upper() for x in group.split(',')]
            picks = [p for p in picks if p in allowed_letters]
            if num_correct == 1 and picks:
                return [picks[0]]
            if num_correct > 1 and len(picks) >= num_correct:
                return picks[:num_correct]

    # Final fallback: scan standalone letters
    tokens = re.findall(r'\b([A-J])\b', upper_text)
    tokens = [t for t in tokens if t in allowed_letters]
    if num_correct == 1:
        return [tokens[0]] if tokens else ["ERROR"]
    # unique order-preserving
    out = []
    for t in tokens:
        if t not in out:
            out.append(t)
        if len(out) == num_correct:
            break
    return out if out else ["ERROR"] * num_correct


def create_chain_of_thought_prompt(question: str, options: str, allowed_letters: List[str], num_correct: int = 1) -> str:
    """Create a prompt for the question using the new concise format with dynamic letter support."""
    if num_correct == 1:
        return f'''Answer the question using one of the given choices.
{question}
{options}

Please begin your response with the exact phrase: "The correct answer is ___." 
Replace the blank with one of the following options: {', '.join(allowed_letters)}.
This line should contain only the final answer. 
Then, in a new paragraph, provide a brief explanation justifying why this answer is correct.
Do not include any introductory phrases, restatements of the question, or additional formatting.'''
    else:
        return f'''Answer the question using {num_correct} of the given choices.
{question}
{options}

Please begin your response with the exact phrase: "The correct answers are ___." 
Replace the blank with {num_correct} of the following options: {', '.join(allowed_letters)}.
Separate multiple answers with commas (e.g., "C,D" or "A,B").
This line should contain only the final answers. 
Then, in a new paragraph, provide a brief explanation justifying why these answers are correct.
Do not include any introductory phrases, restatements of the question, or additional formatting.'''


def call_gemini_with_retry(model: genai.GenerativeModel, prompt: str) -> str:
    """Call Gemini with retries and backoff. Returns response text or empty string on failure."""
    last_err = None
    for delay in [0.0] + RETRY_BACKOFF_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "candidate_count": 1,
                    "top_p": 0.9,
                },
            )
            return getattr(resp, "text", "") or ""
        except Exception as e:
            last_err = e
    print(f"  Gemini call failed after retries: {last_err}")
    return ""


def get_gemini_response(model: genai.GenerativeModel, prompt: str, allowed_letters: List[str], num_runs: int = 3, num_correct: int = 1) -> Tuple[List[str], str, List[List[str]], bool]:
    """Get response from Gemini model with self-consistency checking through multiple runs."""
    try:
        all_answers: List[List[str]] = []
        full_responses: List[str] = []
        
        for run in range(num_runs):
            full_response = call_gemini_with_retry(model, prompt)
            full_responses.append(full_response)

            picks = parse_model_answer(full_response, allowed_letters, num_correct)
            all_answers.append(picks)
            print(f"  Run {run+1}: Answer = {picks}")
            
            time.sleep(PER_RUN_DELAY_SECONDS)
                
        if num_correct == 1:
            consistent = all(a == all_answers[0] for a in all_answers)
        else:
            consistent = all(set(a) == set(all_answers[0]) for a in all_answers)
        
        print(f"  Answers across {num_runs} runs: {all_answers}")
        print(f"  Self-consistency: {consistent}")
        
        if num_correct == 1:
            flat_answers = [a[0] for a in all_answers if a and a[0] != "ERROR"]
            most_common_answer = Counter(flat_answers).most_common(1)[0][0] if flat_answers else "ERROR"
            final_answers = [most_common_answer]
        else:
            answer_tuples = [tuple(sorted(a)) for a in all_answers if a and "ERROR" not in a]
            if answer_tuples:
                most_common_tuple = Counter(answer_tuples).most_common(1)[0][0]
                final_answers = list(most_common_tuple)
            else:
                final_answers = ["ERROR"] * num_correct
        
        combined_response = (
            f"SELF-CONSISTENCY RESULTS:\n"
            f"Runs: {num_runs}\n"
            f"Answers: {all_answers}\n"
            f"Consistent: {consistent}\n"
            f"Majority Answer: {final_answers}\n\n"
            + ("\n\n===RESPONSE #1===\n\n" + (full_responses[0] if full_responses else ""))
        )
        
        return final_answers, combined_response, all_answers, consistent
    except Exception as e:
        print(f"  Error getting Gemini response: {str(e)}")
        return ["ERROR"] * num_correct, f"Error occurred: {str(e)}", [["ERROR"] * num_correct], False


def process_question(model: genai.GenerativeModel, row: pd.Series, num_runs: int = 3) -> Dict:
    """Process a single question and get response from Gemini with self-consistency checking."""
    question = row['Generated Question']
    question_id = row['ID']
    options = row['Choices']
    correct_answer = row['Answer']
    
    difficulty = detect_difficulty_from_options(options)
    allowed_letters = get_letters_for_difficulty(difficulty)
    num_correct = QUESTION_CONFIG[difficulty]["number_of_correct_answer"]
    
    prompt = create_chain_of_thought_prompt(question, options, allowed_letters, num_correct)
    
    print(f"Processing question {question_id} (difficulty: {difficulty}, num_correct: {num_correct})")
    
    gemini_answers, gemini_full_response, all_answers, is_consistent = get_gemini_response(
        model, prompt, allowed_letters, num_runs=num_runs, num_correct=num_correct
    )
    
    correct_answers = extract_answer_letters(correct_answer, allowed_letters, num_correct)
    
    if num_correct == 1:
        match = gemini_answers == correct_answers
    else:
        match = set(gemini_answers) == set(correct_answers)
    
    print(f"  Correct Answer(s): {correct_answers}")
    print(f"  Gemini Answer(s): {gemini_answers}")
    print(f"  Match: {match}")
    
    return {
        'ID': question_id,
        'Question': question,
        'Options': options,
        'Correct_Answer': correct_answer,
        'Correct_Answers_Parsed': ','.join(correct_answers),
        'Gemini_Answer': ','.join(gemini_answers),
        'Full_Response': gemini_full_response,
        'Match': match,
        'Self_Consistent': 'Yes' if is_consistent else 'No',
        'Difficulty': difficulty,
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
    import argparse

    parser = argparse.ArgumentParser(description="Gemini MCQ evaluation")
    parser.add_argument("--file", dest="data_file", default=None,
                         help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv.")
    parser.add_argument("--difficulty", dest="difficulty", default=None,
                         choices=["easy", "medium", "hard"],
                         help="easy=Config 1, medium=Config 2, hard=Config 3. "
                              "Auto-detected from --file if omitted.")
    args = parser.parse_args()

    # Configure Gemini from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    genai.configure(api_key=api_key)

    # Build model once (free tier-friendly model)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    print(f"1. Gemini configured: {model_name}")

    # Read data
    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE.get(args.difficulty, DIFFICULTY_TO_DATA_FILE["easy"])
    df = pd.read_csv(data_file)
    print(f"2. Data loaded: {len(df)} questions from {data_file}")

    difficulty = args.difficulty or detect_difficulty_from_options(
        df.iloc[0]["Choices"] if not df.empty else ""
    )

    # Create results directory
    results_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[difficulty])
    os.makedirs(results_dir, exist_ok=True)
    print(f"3. Results directory: {results_dir} (difficulty: {difficulty})")

    # Process questions
    results = []
    num_runs = 3
    
    print(f"4. Processing {len(df)} questions with {num_runs} runs each...")
    
    for index, row in df.iterrows():
        print(f"\n--- Question {index + 1}/{len(df)} ---")
        result = process_question(model, row, num_runs)
        results.append(result)
        
        # Delay between questions to reduce error rate limiting
        time.sleep(PER_QUESTION_DELAY_SECONDS)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    accuracy = results_df['Match'].mean() * 100 if not results_df.empty else 0.0
    self_consistency = results_df['Is_Consistent'].mean() * 100 if not results_df.empty else 0.0
    
    difficulty_counts = results_df['Difficulty'].value_counts() if not results_df.empty else {}
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Self-consistency: {self_consistency:.2f}%")
    print(f"Difficulty distribution: {dict(difficulty_counts) if hasattr(difficulty_counts, 'to_dict') else difficulty_counts}")
    
    # Save results
    output_file = os.path.join(results_dir, f'gemini_results_{difficulty}_with_consistency.xlsx')
    try:
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        csv_path = os.path.splitext(output_file)[0] + '.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Excel write failed ({e}); saved CSV to: {csv_path}")
    
    # Save summary
    summary = {
        'difficulty': difficulty,
        'total_questions': len(df),
        'accuracy': accuracy,
        'self_consistency': self_consistency,
        'detected_difficulties': dict(difficulty_counts) if hasattr(difficulty_counts, 'to_dict') else difficulty_counts
    }
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(results_dir, f'gemini_summary_{difficulty}.xlsx')
    try:
        summary_df.to_excel(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
    except Exception as e:
        summary_csv = os.path.splitext(summary_file)[0] + '.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"Excel write failed ({e}); saved CSV to: {summary_csv}")
    
    print("=== EVALUATION COMPLETE ===")
