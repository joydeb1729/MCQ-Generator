import os
import json
import pandas as pd 
import traceback
from dotenv import load_dotenv

from src.mcqgen.utils import read_file, get_table_data, create_df
from src.mcqgen.mcq_generator import generate_evaluate_chain
from src.mcqgen.logger import logging 

from langchain.callbacks import get_openai_callback
import streamlit as st

st.set_page_config(layout="wide")


load_dotenv()

# Load the response format template
with open('D:\\codes\\MCQ-Generator\\response.json', 'r') as file:
    response_json = json.load(file)

st.title("MCQ Generator Application")

with st.form('user_inputs'):
    upload_files = st.file_uploader("Upload PDF or TEXT file")
    mcq_counts = st.number_input("No. of MCQs", min_value=1, max_value=20, placeholder='5')
    subject = st.text_input("Subject of Quiz", max_chars=30, placeholder='Object Oriented Programming')
    tone = st.text_input('Complexity of Quiz', max_chars=10, placeholder='Simple')
    button = st.form_submit_button('Create Quiz')
    
    if button and upload_files is not None and mcq_counts and subject and tone:
        with st.spinner('Loading...'):
            try:
                text = read_file(upload_files)
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_counts,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(response_json)
                        }
                    )
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Input Tokens: {cb.prompt_tokens}")
                    print(f"Output Tokens: {cb.completion_tokens}")
                    print(f"Total Costs: {cb.total_cost}")

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("An error occurred during generation.")
            else:
                if isinstance(response, dict):
                    reviewed_quiz = response.get('reviewed_quiz')
                    if reviewed_quiz:
                        try:
                            quiz_dict = get_table_data(reviewed_quiz)
                            df = create_df(quiz_dict)
                            st.subheader("MCQ Table")
                            st.table(df)
                        except Exception as parse_err:
                            st.error("Failed to parse MCQs into table format.")
                            st.text(str(parse_err))
                        
                        st.subheader("Expert Review")
                        st.text_area(label='Review', value=response.get('review', 'No review available.'), height=120)
                    else:
                        st.error("No reviewed quiz data found in response.")
                else:
                    st.write("Unexpected response format:")
                    st.write(response)
