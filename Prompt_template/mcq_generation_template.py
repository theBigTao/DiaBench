CONFIG1_PROMPT = """
You are a medical education assistant specializing in diabetes patient education.  
Your task is to transform a patient-oriented Q&A pair into a multiple-choice question.

### Requirements
- Reading Level: Patient-friendly (9th–12th grade).
- Number of Options: 4 (A-D).
- Number of Correct Answers: 1 (Must be exactly one in most cases).
- Incorrect answers (distractors) must be plausible but clearly wrong.
- Focus on key concepts from the original answer, but you may use general medical knowledge for clarity.
- Only one correct answer should be explicitly indicated.

### Input:
Original Question: {original_question}
Original Answer: {original_answer}

### Output (match exactly)
- Question: <stem here>
- Options:
  {Options}
  
- Answer: <LETTER ONLY>

The correct answer is <LETTER>.
<One or two sentences justifying the choice.>
"""

CONFIG2_PROMPT = """
You are a medical education assistant specializing in diabetes patient education.
Your task is to transform a patient-oriented Q&A pair into a multiple-choice question.

### Requirements
- Reading Level: undergraduate level medical literacy
- Number of Options: 7 (A-G).
- Number of Correct Answers: 1 (Must be exactly one in most cases).
- Incorrect answers (distractors) must be plausible but clearly wrong.
- Focus on key concepts from the original answer, but you may use general medical knowledge for clarity.
- Only one correct answer should be explicitly indicated.

### Input:
Original Question: {original_question}
Original Answer: {original_answer}
  
### Output (match exactly)
- Question: <stem here>
- Options:
  {Options}

- Answer: <LETTER ONLY>

The correct answer is <LETTER>.
<One or two sentences justifying the choice.>
"""

CONFIG3_PROMPT = """
You are an advanced medical education professional specializing in diabetes patient education.  
Your task is to transform a patient-oriented Q&A pair into a multiple-choice question so that you could understand how reliably you staff member might be helping answer questions from their diabetes patients.

### Requirements
- Reading Level: early graduate medical literacy
- Cognitive Level: requires application or analysis, not just recall
- Number of Options: 10 (A-J).
- Number of Correct Answers: 2 (Must be exactly 2).
- Incorrect answers (distractors) must be plausible and based on common misconceptions or clinically adjacent facts.
- Focus on key concepts from the original answer, but integrate broader multi-step clinical reasoning when appropriate.
- Contextual clues or lab values may be introduced to raise complexity.
- Only TWO correct answer should be explicitly indicated.

### Input:
Original Question: {original_question}
Original Answer: {original_answer}


### OUTPUT (match exactly)
- Question: <stem here>
- Options:
  {Options}

- Answer: <LETTER ONLY>

The correct answer is <LETTER>.
<One or two sentences justifying the choice.>
"""
