import pathlib
import textwrap
from dotenv import load_dotenv
import os

from IPython.display import display
from IPython.display import Markdown
import openai
from prettytable import PrettyTable

import PyPDF2

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re

load_dotenv()

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출하는 함수
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def translate_text(text, target_language="en"):
    """
    OpenAI API를 사용하여 텍스트를 번역하는 함수
    :param text: 번역할 원본 텍스트
    :param target_language: 번역 대상 언어 (기본값: 영어)
    :return: 번역된 텍스트
    """
    # OpenAI의 모델에 따라 요청 크기 제한이 있으므로 텍스트를 분할할 필요가 있을 수 있습니다.
    # 여기서는 간단히 전체 텍스트를 한 번에 요청하는 예시를 보여드립니다.
    # 큰 텍스트의 경우, 적절히 분할하여 요청하세요.

    prompt = f"다음 텍스트를 {target_language}로 번역해 주세요:\n\n{text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 사용 가능한 모델로 변경 가능
            messages=[
                {"role": "system", "content": "당신은 유능한 번역가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,  # 필요에 따라 조정
            temperature=0.3  # 창의성 조절
        )
        translated_text = response.choices[0].message['content'].strip()
        return translated_text
    except openai.error.OpenAIError as e:
        print(f"OpenAI API 오류 발생: {e}")
        return None

dir_abs_path=os.path.abspath("../pdf")
print(dir_abs_path)

file_lists=os.listdir(dir_abs_path)
print(file_lists)

len(file_lists)

file_list=[]

for i in range(len(file_lists)):
    if os.path.splitext(file_lists[i])[1]==".pdf":
        file_list.append(file_lists[i])
        
print(file_list)

len(file_list)

cnt=0

for i in range(len(file_list)):
    pdfReader=PyPDF2.PdfReader(f"../pdf/{file_list[i]}")
    print(" No. Of Pages : ",len(pdfReader.pages))
    if len(pdfReader.pages) >= 15:
        cnt=cnt+1
        
print(cnt)

for i in range(len(file_list)):
    file_path=f"../pdf/{file_list[i]}"
    text=extract_text_from_pdf(file_path)
    
    if not os.path.exists("../txt/"):
        os.makedirs("../txt")
        
    with open(f"../txt/{file_list[i]}.txt", 'w') as f:
        f.write(text)
        
        
for i in range(len(file_list)):
    with open(f"../txt/{file_list[i]}.txt", "r") as f:
        text = f.read()
        
    replaced_text = re.sub(r'Page\s*\d+|\s*\d+\s*\n', '', text)
    replaced_text = re.sub(r'-\n', '', replaced_text)
    replaced_text = re.sub(r'\s+', ' ', replaced_text).strip()

    replaced_text
    
    if not os.path.exists("../re_txt/"):
        os.makedirs("../re_txt")
    
    with open(f"../re_txt/{file_list[i]}_cleaned.txt", "w") as f:
        f.write(replaced_text)

for i in range(len(file_list)):
    with open(f"../re_txt/{file_list[i]}_cleaned.txt", "r") as f:
        replaced_text = f.read()
        
    sections = ["ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHOD", "EXPERIMENT", "CONCLUSION"]
    section_data = {}
    
    for j in range(len(sections)-1):
        pattern = f"{sections[j]}(.*?){sections[j+1]}"
        match = re.search(pattern, replaced_text, re.S | re.I)
        if match:
            section_data[sections[j]] = match.group(1).strip()
    
    last_section = sections[-1]
    pattern = f"{last_section}(.*?)$"
    match = re.search(pattern, replaced_text, re.S | re.I)
    
    if match:
        section_data[last_section] = match.group(1).strip()
    
    print(f"--- {file_list[i]} ---")

    for sec, content in section_data.items():
        print(f"--- {sec} ---\n{content[:500]}...\n")
    
    print("-"*25)
    
for i in range(len(file_list)):
    with open(f"../re_txt/{file_list[i]}_cleaned.txt", "r") as f:
        replaced_text = f.read()
        
    sections = ["ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHOD", "EXPERIMENT", "CONCLUSION"]
    section_data = {}
    
    for j in range(len(sections)-1):
        pattern = f"{sections[j]}(.*?){sections[j+1]}"
        match = re.search(pattern, replaced_text, re.S | re.I)
        if match:
            section_data[sections[j]] = match.group(1).strip()
    
    last_section = sections[-1]
    pattern = f"{last_section}(.*?)$"
    match = re.search(pattern, replaced_text, re.S | re.I)
    
    if match:
        section_data[last_section] = match.group(1).strip()
    
    print(file_list[i])

    if not os.path.exists("../translated_txt/"):
        os.makedirs("../translated_txt")
        
    with open(f'../translated_txt/{file_list[i]}_translated.txt', 'w') as f:
        f.write("")
        
    for sec, content in section_data.items():
        translated_text=translate_text(content, target_language="ko")
        with open(f"../translated_txt/{file_list[i]}_translated.txt", 'a') as f:
            f.write(f"--- {sec} ---")
            f.write("\n")
            f.write(translated_text)
            f.write("\n")