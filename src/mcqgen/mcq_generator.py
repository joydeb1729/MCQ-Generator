import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import PyPDF2

from src.mcqgen.utils import read_file, get_table_data
from src.mcqgen.logger import logging 

from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback

load_dotenv()
API_KEY = os.getenv('OPENROUTER_API_KEY2')

llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="shisa-ai/shisa-v2-llama3.3-70b:free",
    temperature=0.5,
)

# ----------- TEMPLATE 1: Generation -------------
TEMPLATE1 = """
You are an expert question setter with deep knowledge of the subject: {subject}. 
Generate {number} multiple choice questions (MCQs) from the following text. Use a {tone} tone while creating the questions.

TEXT:
{text}

INSTRUCTIONS:
- Each question should have exactly 4 options: A, B, C, and D.
- Only one option should be correct.
- Vary difficulty from easy to hard and cover different parts of the text.
- Ensure relevance and factual accuracy.
- Do not include explanations, code fences, or any extra text.

RESPONSE FORMAT should be exactly as:
{response_json} without any extra information
"""

mcq_generation_prompt = PromptTemplate(
    input_variables=['text', 'number', 'subject', 'tone', 'response_json'],
    template=TEMPLATE1
)

generation_chain = LLMChain(
    llm=llm,
    prompt=mcq_generation_prompt,
    output_key="quiz",
    verbose=True
)

# ----------- TEMPLATE 2: Evaluation -------------
TEMPLATE2 = """
You are an expert in English grammar and education, skilled at analyzing and adjusting multiple choice questions (MCQs) for clarity, tone, and student ability.

Subject: {subject}

Below is a JSON array of generated MCQs:
{quiz}

INSTRUCTIONS:
- Check and correct all grammar.
- Ensure each question’s complexity matches the students’ level.
- Maintain relevance to the subject.
- Preserve the specified tone carefully: {tone}.
- If any question is too complex, unclear, or off-tone, revise it while keeping its original intent.

RESPONSE FORMAT should be exactly same as {response_json} without any extra text:
"""

mcq_evaluation_prompt = PromptTemplate(
    input_variables=['quiz', 'subject', 'tone', 'response_json'],
    template=TEMPLATE2
)

recheck_chain = LLMChain(
    llm=llm,
    prompt=mcq_evaluation_prompt,
    output_key="reviewed_quiz",
    verbose=True
)

# ----------- TEMPLATE 3: Review -------------
TEMPLATE3 = """
You are an experienced education expert evaluating a set of multiple choice questions (MCQs) for effectiveness, clarity, and student readiness.

Subject: {subject}

Below is a JSON array of revised MCQs:
{reviewed_quiz}

INSTRUCTIONS:
- Review the entire set of questions as a whole.
- Provide a concise evaluation (under 100 words) covering:
  - Overall clarity and tone,
  - Suitability of question complexity for the students,
  - Whether students are likely to be able to answer them successfully.

RESPONSE FORMAT:
Return a single plain text paragraph (no JSON, no bullet points, no extra formatting).
"""

mcq_review_prompt = PromptTemplate(
    input_variables=['subject', 'reviewed_quiz'],
    template=TEMPLATE3
)

review_chain = LLMChain(
    llm=llm,
    prompt=mcq_review_prompt,
    output_key="review",
    verbose=True
)

# ----------- Sequential Chain with fix -------------
generate_evaluate_chain = SequentialChain(
    chains=[generation_chain, recheck_chain, review_chain],
    input_variables=['text', 'number', 'subject', 'tone', 'response_json'],
    output_variables=['quiz', 'reviewed_quiz', 'review'], 
    return_all=True,
    verbose=True
)
