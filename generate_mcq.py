import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types
import argparse
from openai import OpenAI
from Prompt_template.mcq_generation_template import (
    CONFIG1_PROMPT,
    CONFIG2_PROMPT,
    CONFIG3_PROMPT,
)
import re

load_dotenv()

# -----------------------------
# Config
# -----------------------------
# Strongly recommended: do NOT hardcode secrets. If you must, at least read from env.


# -----------------------------
# Prompt
# -----------------------------
QUESTION_CONFIG = {
    "CONFIG1": {"number_of_answer_options": 4, "number_of_correct_answer": 1},
    "CONFIG2": {"number_of_answer_options": 7, "number_of_correct_answer": 1},
    "CONFIG3": {"number_of_answer_options": 10, "number_of_correct_answer": 2},
}

PROMPT_OPTIONS = {
    4: {
        "options": """
                A) ...
                B) ...
                C) ...
                D) ...
            """,
    },
    7: {
        "options": """
            A) ...
            B) ...
            C) ...
            D) ...
            E) ...
            F) ...
            G) ...
    """,
    },
    10: {
        "options": """ 
            A) ...
            B) ...
            C) ...
            D) ...
            E) ...
            F) ...
            G) ...
            H) ...
            I) ...
            J) ...
            """,
    },
}
CORRECT_ANSWER = {1: "ONE", 2: "TWO", 3: "THREE"}


def build_prompt(
    question: str,
    answer: str,
    prompt_template: ["CONFIG1", "CONFIG2", "CONFIG3"],
) -> str:
    if prompt_template == "CONFIG1":
        template = CONFIG1_PROMPT
        number_of_answer_options = QUESTION_CONFIG["CONFIG1"][
            "number_of_answer_options"
        ]
    elif prompt_template == "CONFIG2":
        template = CONFIG2_PROMPT
        number_of_answer_options = QUESTION_CONFIG["CONFIG2"][
            "number_of_answer_options"
        ]
    elif prompt_template == "CONFIG3":
        template = CONFIG3_PROMPT
        number_of_answer_options = QUESTION_CONFIG["CONFIG3"][
            "number_of_answer_options"
        ]

    prompt = template.format(
        original_question=str(question).strip(),
        original_answer=str(answer).strip(),
        Options=PROMPT_OPTIONS[number_of_answer_options]["options"],
    )
    return prompt


# -----------------------------
# OpenAI Call
# -----------------------------
def call_LLM(client, prompt: str, LLM: ["gemini", "gpt"]) -> str:
    if LLM == "gemini":
        try:
            response = client.models.generate_content(
                contents=prompt,
                model="gemini-2.5-flash",
                # model="gemini-2.5-pro",
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text
        except Exception as e:
            print("Gemini error:", e)
            return None

    if LLM == "gpt":
        try:
            resp = client.chat.completions.create(
                model="gpt-5",
                # model="o3",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                reasoning={
                    "effort": "medium"
                },  # Medium is default. Options: minimal, low, medium, and high
                text={
                    "verbosity": "medium"
                },  # Medium is Default. Determine the number of generated tokens. Options: high, medium, or low.
            )
            return resp.choices[0].message.content
        except Exception as e:
            print("OpenAI error:", e)
            return None
    else:
        raise ValueError(f"Unsupported LLM: {LLM}")


# -----------------------------
# Parser (your original logic, lightly tidied)
# -----------------------------
def parse_response_to_components(response: str):
    try:
        if not response:
            return None, None, None
        lines = response.strip().splitlines()

        # Extract question
        q_text = None
        patterns = [
            "- Question:",
            "**Question:**",
            "Question:",
            "Question :",
            "**Question**:",
            "#### Question:",
            "#### Generated Question:",
        ]
        for line in lines:
            for p in patterns:
                if p in line:
                    q_text = line.split(p, 1)[1].strip()
                    break
            if q_text:
                break
        if not q_text:
            # fallback: first line with a question mark long enough
            q_cands = [l.strip() for l in lines if "?" in l and len(l) > 20]
            if q_cands:
                q_text = q_cands[0]

        if q_text:
            for junk in ['"', "*", "_", "####"]:
                q_text = q_text.replace(junk, "")
            q_text = q_text.strip("\"'")

        # Extract options A–J
        option_lines = []
        for line in lines:
            s = line.strip()
            if any(
                s.startswith(f"{ch})")
                or s.startswith(f"{ch}.")
                or s.startswith(f"**{ch}**")
                for ch in "ABCDEFGHIJ"
            ):
                option_lines.append(s)
        unique = []
        seen = set()
        for opt in option_lines:
            letter = opt[0].upper()
            if letter in "ABCDEFGHIJ" and letter not in seen:
                unique.append(opt.replace("**", "").replace("*", "").replace("_", ""))
                seen.add(letter)
        if len(unique) != 10 and len(unique) > 10:
            unique = unique[:10]

        # Extract answer letters as a string, e.g., "A, F"
        answer_str = None
        ans_patterns = [
            "- Answer:",
            "**Answer:**",
            "Answer:",
            "Correct Answer:",
            "**Answer**:",
            "#### Answer:",
        ]
        for p in ans_patterns:
            for line in lines:
                if p in line:
                    part = line.split(p, 1)[1].strip()
                    # Find all capital letters A–J
                    letters = re.findall(r"\b([A-J])\b", part)
                    if letters:
                        answer_str = ", ".join(letters)
                        break
            if answer_str:
                break
        if not answer_str:
            # generic fallback: look for lines mentioning answer/correct and extract all letters
            for line in lines:
                lo = line.lower()
                if "answer" in lo or "correct" in lo:
                    letters = re.findall(r"\b([A-J])\b", line)
                    if letters:
                        answer_str = ", ".join(letters)
                        break
        if not answer_str and unique:
            answer_str = unique[0][0]

        if q_text and unique and answer_str:
            return q_text, unique, answer_str
        return None, None, None

    except Exception as e:
        print("Parsing error:", e)
        return None, None, None


# -----------------------------
# IO helpers
# -----------------------------
def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # trim/standardize column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # map acceptable variants -> canonical names
    col_map = {}
    # Question
    for cand in ["Question", "Original Question", "Prompt", "Q"]:
        if cand in df.columns:
            col_map[cand] = "Question"
            break
    # Answer
    for cand in ["Answer", "Original Answer", "A"]:
        if cand in df.columns:
            col_map[cand] = "Answer"
            break

    df = df.rename(columns=col_map)
    required = ["Question", "Answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Columns found: {df.columns.tolist()}"
        )

    # basic cleanup
    df = df.dropna(subset=["Question", "Answer"])
    df = df[df["Question"].astype(str).str.strip() != ""]
    df = df[df["Answer"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def build_llm_client(llm: str):
    assert llm in ["gemini", "gpt"], "Invalid LLM specified"
    print(f"Building LLM client for: {llm}")
    if llm == "gemini":
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=GEMINI_API_KEY)
    elif llm == "gpt":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# Main build
# -----------------------------
def build_mcq_df(
    client,
    prompt_template,
    df: pd.DataFrame,
    llm: str,
    max_retries=3,
    retry_delay=2,
) -> pd.DataFrame:
    # df = df.iloc[:3]
    records = []
    raw_response = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating MCQs"):
        question, answer = row["Question"], row["Answer"]
        prompt = build_prompt(
            question,
            answer,
            prompt_template=prompt_template,
        )
        for attempt in range(1, max_retries + 1):
            resp = call_LLM(client, prompt, llm)
            if not resp:
                time.sleep(retry_delay)
                continue
            raw_response.append(resp)
            new_q, choices, correct = parse_response_to_components(resp)
            if new_q and choices and correct:
                records.append(
                    {
                        "ID": idx + 1,
                        "Original Question": question,
                        "Original Answer": answer,
                        "Generated Question": new_q,
                        "Choices": "\n".join(choices),
                        "Answer": correct,
                    }
                )
                break
            time.sleep(retry_delay)
        time.sleep(0.8)  # small buffer
    with open("output.txt", "w") as f:
        f.write("=================\n".join(raw_response))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate MCQs using LLMs")
    parser.add_argument(
        "--llm", choices=["gemini", "gpt"], default="gemini", help="Which LLM to use"
    )
    parser.add_argument(
        "--prompt_template",
        choices=["CONFIG1", "CONFIG2", "CONFIG3"],
        default="CONFIG1",
        help="Prompt template to use",
    )
    # parser.add_argument(
    #     "--num-options", type=int, default=4, help="Number of answer options"
    # )
    # parser.add_argument(
    #     "--num-correct", type=int, default=1, help="Number of correct answers"
    # )
    parser.add_argument(
        "--input-file",
        type=str,
        default="original_questions.csv",
        help="Input file (.csv or .xlsx)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="New_Numerics", help="Output directory"
    )
    args = parser.parse_args()
    if args.prompt_template not in QUESTION_CONFIG:
        print(f"Invalid prompt template: {args.prompt_template}")
        print("Valid options are:", list(QUESTION_CONFIG.keys()))
        raise ValueError("Invalid prompt template specified")
    print("================Args===================")
    print("LLM:", args.llm)
    print("Prompt Template:", args.prompt_template)
    print("Input file:", args.input_file)
    print("Output directory:", args.output_dir)
    print("==================================")
    INPUT_FILE = args.input_file
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    load_dotenv()
    client = build_llm_client(args.llm)
    try:
        df_raw = read_any_table(INPUT_FILE)
        print(f"Loaded {len(df_raw)} rows from {INPUT_FILE}")
        print("Columns in DataFrame:", df_raw.columns.tolist())

        df = normalize_columns(df_raw)
        print(f"Normalized to {len(df)} usable rows.")

        print("Processing MCQ generation…")
        mcq_df = build_mcq_df(
            client=client,
            df=df,
            prompt_template=args.prompt_template,
            max_retries=3,
            retry_delay=2,
            llm=args.llm,
        )

        out_path = os.path.join(
            OUTPUT_DIR, f"MCQs_{args.llm}_{args.prompt_template}.csv"
        )
        mcq_df.to_csv(out_path, index=False)
        print(f"Saved {len(mcq_df)} results to {out_path}")

    except Exception as e:
        print(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
