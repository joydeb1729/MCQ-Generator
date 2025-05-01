import os
import json
import pandas as pd
import traceback
import PyPDF2

def read_file(file):
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extract_text()
            return text
        except Exception as e:
            raise Exception('Error occurred while reading the PDF!')
    
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    
    else:
        raise Exception('Unsupported file format! Please provide a .txt or .pdf file.')

def get_table_data(quiz_str):
    try:
        if quiz_str.startswith("```json"):
            quiz_str = quiz_str.replace("```json", "").replace("```", "").strip()
        
        quiz_dict = json.loads(quiz_str)
        return quiz_dict
    
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False

def create_df(data):
    df = pd.DataFrame.from_dict(data, orient='index')
    
    if 'options' in df.columns:
        options_df = df['options'].apply(pd.Series)
        df = df.drop(columns='options').join(options_df)
    
    if 'mcq' in df.columns:
        df.rename(columns={'mcq': 'Question'}, inplace=True)

    if 'no' in df.columns:
        df.drop(columns='no', inplace=True)

    if 'correct' in df.columns:
        correct = df.pop('correct')
        df['correct'] = correct

    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df
