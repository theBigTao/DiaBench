import os
import time
import re
import json
import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
import tempfile
import uuid

import pandas as pd
from dotenv import load_dotenv
import argparse

from common import (
    QUESTION_CONFIG,
    get_letters_for_difficulty,
    detect_difficulty_from_options,
    extract_answer_letters,
    save_df_excel_or_csv,
)

print("=== DEEPSEEK-R1 EVALUATION PIPELINE (TOGETHER AI BATCH MODE) ===")

# -------------------------
# Batch API configuration
# -------------------------
BATCH_CHECK_INTERVAL = 60  # seconds
MAX_BATCH_WAIT_TIME = 24 * 60 * 60  # 24 hours in seconds
BATCH_FILE_SIZE_LIMIT = 100 * 1024 * 1024  # 100MB
MAX_REQUESTS_PER_BATCH = 50000

# Checkpoint paths for batch processing
BATCH_PROGRESS_PATH = os.path.join("results", "deepseek_batch_progress.json")
BATCH_RESULTS_PATH = os.path.join("results", "deepseek_batch_results.json")

# -------------------------
# Helpers (provider-specific — DeepSeek-R1 batch request/response handling)
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


def build_together_client():
    """Build Together AI client."""
    try:
        load_dotenv()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
        print("Using environment variables directly...")
    
    try:
        from together import Together
    except Exception as e:
        raise RuntimeError("together package not installed. Try: pip install together") from e

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TOGETHER_API_KEY in environment/.env")

    client = Together(api_key=api_key)
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
    return client, model_name


def create_batch_file(questions_data: List[Dict], model_name: str, num_runs: int = 3) -> str:
    """Create a JSONL batch file for Together AI batch API."""
    batch_requests = []
    
    for i, question_data in enumerate(questions_data):
        # Handle different column names in the dataset
        question_id = question_data.get('ID', f'q{i+1}')
        if pd.isna(question_id):
            question_id = f'q{i+1}'
        
        # Try different possible column names
        question = question_data.get('Generated Question', '') or question_data.get('1st Question', '')
        options = question_data.get('Choices', '') or question_data.get('1st Choices', '')
        correct_answer = question_data.get('Answer', '') or question_data.get('1st Answer', '')
        
        # Auto-detect difficulty
        diff = detect_difficulty_from_options(options)
        allowed = get_letters_for_difficulty(diff)
        num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]
        
        # Create multiple runs for self-consistency
        for run in range(num_runs):
            prompt = create_eval_prompt(question, options, allowed, num_correct)
            
            request_data = {
                "custom_id": f"q{question_id}_run{run}",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.6
                }
            }
            batch_requests.append(request_data)
    
    # Create temporary JSONL file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    for request in batch_requests:
        temp_file.write(json.dumps(request) + '\n')
    
    temp_file.close()
    return temp_file.name


def upload_batch_file(client, file_path: str) -> str:
    """Upload batch file to Together AI."""
    try:
        # Pass the file path directly, not a file object
        file_response = client.files.upload(file=file_path, purpose="batch-api")
        return file_response.id
    except Exception as e:
        raise RuntimeError(f"Failed to upload batch file: {e}")


def create_batch_job(client, file_id: str) -> str:
    """Create a batch job."""
    try:
        batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")
        return batch.id
    except Exception as e:
        raise RuntimeError(f"Failed to create batch job: {e}")


def monitor_batch_job(client, batch_id: str) -> Dict:
    """Monitor batch job until completion."""
    start_time = time.time()
    
    while True:
        try:
            batch_status = client.batches.get_batch(batch_id)
            print(f"Batch status: {batch_status.status}")
            
            if batch_status.status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                return batch_status
            
            if time.time() - start_time > MAX_BATCH_WAIT_TIME:
                raise RuntimeError("Batch job timed out after 24 hours")
            
            time.sleep(BATCH_CHECK_INTERVAL)
            
        except Exception as e:
            print(f"Error monitoring batch: {e}")
            time.sleep(BATCH_CHECK_INTERVAL)


def download_batch_results(client, output_file_id: str, output_path: str) -> str:
    """Download batch results."""
    try:
        # Use the original method that was working
        client.files.retrieve_content(id=output_file_id, output=output_path)
        return output_path
    except Exception as e:
        # If the file was already downloaded, try to find it
        if os.path.exists(output_path):
            print(f"Results file found at: {output_path}")
            return output_path
        raise RuntimeError(f"Failed to download batch results: {e}")


def parse_batch_results(results_file: str, questions_data: List[Dict], num_runs: int = 3) -> List[Dict]:
    """Parse batch results and combine with original question data."""
    results = []
    
    # Load batch results with proper encoding
    batch_results = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get('custom_id', '')
                batch_results[custom_id] = result
    
    # Process each question
    for i, question_data in enumerate(questions_data):
        # Handle different column names in the dataset
        question_id = question_data.get('ID', f'q{i+1}')
        if pd.isna(question_id):
            question_id = f'q{i+1}'
        
        # Try different possible column names
        question = question_data.get('Generated Question', '') or question_data.get('1st Question', '')
        options = question_data.get('Choices', '') or question_data.get('1st Choices', '')
        correct_answer = question_data.get('Answer', '') or question_data.get('1st Answer', '')
        
        # Auto-detect difficulty
        diff = detect_difficulty_from_options(options)
        allowed = get_letters_for_difficulty(diff)
        num_correct = QUESTION_CONFIG[diff]["number_of_correct_answer"]
        
        # Collect responses for this question
        all_responses = []
        all_answers = []
        
        for run in range(num_runs):
            custom_id = f"q{question_id}_run{run}"
            if custom_id in batch_results:
                response_data = batch_results[custom_id]
                if 'response' in response_data and 'body' in response_data['response']:
                    response_text = response_data['response']['body'].get('choices', [{}])[0].get('message', {}).get('content', '')
                    all_responses.append(response_text)
                    
                    # Parse answer
                    picks = parse_response_for_answers(response_text, allowed, num_correct)
                    all_answers.append(picks)
                else:
                    all_responses.append("ERROR: No response")
                    all_answers.append([])
            else:
                all_responses.append("ERROR: Missing response")
                all_answers.append([])
        
        # Calculate consistency and final answer
        if num_correct == 1:
            non_empty = [a for a in all_answers if a]
            is_consistent = len(set(tuple(x) for x in non_empty)) == 1 if non_empty else False
            flat = [a[0] for a in non_empty]
            final = [Counter(flat).most_common(1)[0][0]] if flat else []
        else:
            sets = [tuple(sorted(a)) for a in all_answers if a]
            is_consistent = len(set(sets)) == 1 if sets else False
            final = list(Counter(sets).most_common(1)[0][0]) if sets else []
        
        # Get gold standard answer
        gold = extract_answer_letters(correct_answer, allowed, num_correct)
        match = (final == gold) if num_correct == 1 else (set(final) == set(gold))
        
        # Create detailed response showing all runs
        combined_text = (
            "SELF-CONSISTENCY RESULTS\n"
            f"Runs: {num_runs}\n"
            f"Answers: {all_answers}\n"
            f"Consistent: {is_consistent}\n\n"
        )
        
        # Add all individual responses
        for i, response in enumerate(all_responses):
            combined_text += f"=== RESPONSE #{i+1} ===\n"
            combined_text += f"{response}\n\n"
        
        result = {
            'ID': question_id,
            'Question': question,
            'Options': options,
            'Correct_Answer': correct_answer,
            'Correct_Answers_Parsed': ','.join(gold),
            'DeepSeek_Answer': ','.join(final),
            'Full_Response': combined_text,
            'Match': match,
            'Self_Consistent': 'Yes' if is_consistent else 'No',
            'Difficulty': diff,
            'Num_Correct': num_correct,
            'All_Answers': str(all_answers),
            'Is_Consistent': is_consistent
        }
        results.append(result)
    
    return results


def parse_response_for_answers(response_text: str, allowed_letters: List[str], num_correct: int = 1) -> List[str]:
    """Parse response text to extract answer letters."""
    if not response_text or response_text.startswith("ERROR"):
        return []
    
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
        return [pick] if pick else []
    else:
        pick = []
        # More specific patterns for multiple answers - prioritize "and" format first
        patterns = [
            r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*([A-J](?:\s+AND\s+[A-J])*)',
            r'THE\s+ANSWERS\s+ARE\s*([A-J](?:\s+AND\s+[A-J])*)',
            r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*\[([A-J](?:\s*,\s*[A-J])*)\]',
            r'THE\s+CORRECT\s+ANSWERS\s+ARE\s*([A-J](?:\s*,\s*[A-J])*)',
            r'THE\s+ANSWERS\s+ARE\s*\[([A-J](?:\s*,\s*[A-J])*)\]',
            r'THE\s+ANSWERS\s+ARE\s*([A-J](?:\s*,\s*[A-J])*)',
            r'\[([A-J](?:\s*,\s*[A-J])*)\]',
            r'\(([A-J](?:\s*,\s*[A-J])*)\)',
        ]
        
        for pat in patterns:
            m = re.search(pat, txt_up)
            if m:
                # Extract letters from the matched group
                match_text = m.group(1).upper()
                # Use word boundaries to avoid matching letters within words like "AND"
                letters = re.findall(r'\b[A-J]\b', match_text)
                pick = [l for l in letters if l in allowed_letters][:num_correct]
                if len(pick) == num_correct:
                    break
        
        # If no specific pattern matched, try to find the last occurrence of answer format
        if not pick:
            # Look for "A and B" or "A, B" patterns at the end of the text
            end_patterns = [
                r'([A-J](?:\s+AND\s+[A-J])*)\s*\.?\s*$',
                r'([A-J](?:\s*,\s*[A-J])*)\s*\.?\s*$',
            ]
            for pat in end_patterns:
                m = re.search(pat, txt_up)
                if m:
                    match_text = m.group(1).upper()
                    letters = re.findall(r'\b[A-J]\b', match_text)
                    pick = [l for l in letters if l in allowed_letters][:num_correct]
                    if len(pick) == num_correct:
                        break
        
        return pick if pick else []


def save_batch_progress(batch_id: str, file_id: str, questions_count: int, num_runs: int, model_name: str):
    """Save batch job progress for resuming."""
    progress_data = {
        "batch_id": batch_id,
        "file_id": file_id,
        "questions_count": questions_count,
        "num_runs": num_runs,
        "model_name": model_name,
        "start_time": datetime.datetime.now().isoformat(),
        "status": "IN_PROGRESS"
    }
    
    os.makedirs(os.path.dirname(BATCH_PROGRESS_PATH), exist_ok=True)
    with open(BATCH_PROGRESS_PATH, 'w') as f:
        json.dump(progress_data, f, indent=2)
    print(f"Batch progress saved: {BATCH_PROGRESS_PATH}")


def load_batch_progress():
    """Load existing batch job progress."""
    if os.path.exists(BATCH_PROGRESS_PATH):
        with open(BATCH_PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return None


def process_questions_batch(client, model_name: str, questions_data: List[Dict], num_runs: int = 3) -> List[Dict]:
    """Process questions using Together AI batch API with progress tracking."""
    print(f"Creating batch file for {len(questions_data)} questions with {num_runs} runs each...")
    
    # Check for existing batch job
    existing_progress = load_batch_progress()
    if existing_progress:
        print(f"Found existing batch job: {existing_progress['batch_id']}")
        print("Checking status...")
        
        try:
            batch_status = client.batches.get_batch(existing_progress['batch_id'])
            print(f"Existing batch status: {batch_status.status}")
            
            if batch_status.status == 'COMPLETED':
                print("Existing batch completed! Downloading results...")
                results_file = f"batch_results_{existing_progress['batch_id']}.jsonl"
                download_batch_results(client, batch_status.output_file_id, results_file)
                results = parse_batch_results(results_file, questions_data, num_runs)
                os.unlink(results_file)
                
                # Clean up progress file
                if os.path.exists(BATCH_PROGRESS_PATH):
                    os.unlink(BATCH_PROGRESS_PATH)
                
                return results
            elif batch_status.status in ['IN_PROGRESS', 'VALIDATING']:
                print("Resuming monitoring of existing batch job...")
                batch_status = monitor_batch_job(client, existing_progress['batch_id'])
                
                if batch_status.status == 'COMPLETED':
                    print("Batch job completed successfully!")
                    results_file = f"batch_results_{existing_progress['batch_id']}.jsonl"
                    download_batch_results(client, batch_status.output_file_id, results_file)
                    results = parse_batch_results(results_file, questions_data, num_runs)
                    os.unlink(results_file)
                    
                    # Clean up progress file
                    if os.path.exists(BATCH_PROGRESS_PATH):
                        os.unlink(BATCH_PROGRESS_PATH)
                    
                    return results
                else:
                    raise RuntimeError(f"Batch job failed with status: {batch_status.status}")
            else:
                print(f"Existing batch job has status: {batch_status.status}. Starting new batch...")
        except Exception as e:
            print(f"Error checking existing batch: {e}. Starting new batch...")
    
    # Create new batch job
    batch_file_path = create_batch_file(questions_data, model_name, num_runs)
    print(f"Batch file created: {batch_file_path}")
    
    try:
        # Upload batch file
        print("Uploading batch file...")
        file_id = upload_batch_file(client, batch_file_path)
        print(f"File uploaded with ID: {file_id}")
        
        # Create batch job
        print("Creating batch job...")
        batch_id = create_batch_job(client, file_id)
        print(f"Batch job created with ID: {batch_id}")
        
        # Save progress
        save_batch_progress(batch_id, file_id, len(questions_data), num_runs, model_name)
        
        # Monitor batch job
        print("Monitoring batch job...")
        batch_status = monitor_batch_job(client, batch_id)
        
        if batch_status.status == 'COMPLETED':
            print("Batch job completed successfully!")
            
            # Download results
            results_file = f"batch_results_{batch_id}.jsonl"
            print(f"Downloading results to {results_file}...")
            download_batch_results(client, batch_status.output_file_id, results_file)
            
            # Parse results
            print("Parsing results...")
            results = parse_batch_results(results_file, questions_data, num_runs)
            
            # Clean up
            os.unlink(results_file)
            if os.path.exists(BATCH_PROGRESS_PATH):
                os.unlink(BATCH_PROGRESS_PATH)
            
            return results
        else:
            raise RuntimeError(f"Batch job failed with status: {batch_status.status}")
    
    finally:
        # Clean up batch file
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)


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
    parser = argparse.ArgumentParser(description="DeepSeek-R1 MCQ evaluation using Together AI batch API")
    parser.add_argument("--file", dest="data_file", default=None,
                       help="Path to MCQs CSV/XLSX. Defaults to ../Generated_MCQs/MCQs_Config{1,2,3}.csv.")
    parser.add_argument("--difficulty", dest="difficulty", default=None,
                       choices=["easy", "medium", "hard"],
                       help="easy=Config 1, medium=Config 2, hard=Config 3. "
                            "Auto-detected from --file if omitted.")
    parser.add_argument("--runs", dest="num_runs", type=int, default=3,
                       help="Number of runs per question for self-consistency")
    parser.add_argument("--limit", dest="limit", type=int, default=None,
                       help="Limit number of questions for testing")
    args = parser.parse_args()

    # Setup
    client, model_name = build_together_client()
    print(f"1. Together AI model: {model_name}")

    # Read data
    data_file = args.data_file or DIFFICULTY_TO_DATA_FILE.get(args.difficulty, DIFFICULTY_TO_DATA_FILE["easy"])
    df = pd.read_csv(data_file) if data_file.lower().endswith('.csv') else pd.read_excel(data_file)
    
    # Limit questions for testing if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"2. Limited to first {len(df)} questions for testing")
    else:
        print(f"2. Data loaded: {len(df)} questions from {data_file}")

    # Difficulty: explicit flag > filename hint > auto-detect from the data itself
    if args.difficulty:
        difficulty = args.difficulty
    elif "medium" in data_file.lower():
        difficulty = "medium"
    elif "hard" in data_file.lower():
        difficulty = "hard"
    elif "easy" in data_file.lower():
        difficulty = "easy"
    else:
        difficulty = detect_difficulty_from_options(df.iloc[0]["Choices"] if not df.empty else "")

    out_dir = os.path.join("results", DIFFICULTY_TO_FOLDER[difficulty])
    os.makedirs(out_dir, exist_ok=True)
    print(f"3. Results dir: {out_dir} (difficulty: {difficulty})")

    num_runs = args.num_runs
    print(f"4. Processing with {num_runs} runs per question using batch API")

    # Convert DataFrame to list of dictionaries
    questions_data = df.to_dict('records')
    
    # Process questions in batch
    print("\n=== STARTING BATCH PROCESSING ===")
    results = process_questions_batch(client, model_name, questions_data, num_runs)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    acc = float(results_df['Match'].mean() * 100) if not results_df.empty else 0.0
    self_cons = float(results_df['Is_Consistent'].mean() * 100) if not results_df.empty else 0.0

    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Self-consistency: {self_cons:.2f}%")

    # Save results in the same format as other LLMs
    out_main = os.path.join(out_dir, f'deepseek_results_{difficulty}_with_consistency.xlsx')
    save_df_excel_or_csv(results_df, out_main)
    
    # Also save checkpoint CSV like other LLMs
    checkpoint_file = os.path.join(out_dir, 'deepseek_results_checkpoint.csv')
    results_df.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved: {checkpoint_file}")

    summary = pd.DataFrame([{
        'difficulty': difficulty,
        'total_questions': len(df),
        'accuracy': acc,
        'self_consistency': self_cons
    }])
    save_df_excel_or_csv(summary, os.path.join(out_dir, 'deepseek_summary.xlsx'))

    print("=== DEEPSEEK-R1 BATCH EVALUATION COMPLETE ===")
