import pathlib
import textwrap
from dotenv import load_dotenv
import os

import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
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

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

client=documentai.DocumentProcessorServiceClient()

project_id = os.getenv('PROJECT_ID')
location = os.getenv('LOCATION')
processor_id=os.getenv('PROCESSOR_ID')

print(project_id, location, processor_id)

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
    with open(f"../pdf/{file_list[i]}", 'rb') as f:
        content=f.read()
        
    request = documentai.types.ProcessRequest(
        name=f"projects/{project_id}/locations/{location}/processors/{processor_id}",
        raw_document=documentai.types.RawDocument(
            content=content,
            mime_type="application/pdf"
        )
    )
    
    result = client.process_document(request=request)
    
    document = result.document
    text=document.text
    
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
        
def detect_language(text: str) -> dict:
    """Detects the text's language."""
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.detect_language(text)

    print(f"Text: {text}")
    print("Confidence: {}".format(result["confidence"]))
    print("Language: {}".format(result["language"]))

    return result
  
for i in range(len(file_list)):
    with open(f"../txt/{file_list[i]}.txt", "r") as f:
        text = f.read()
        
    detect_language(text=text)
    
def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result

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
        translated_text=translate_text(target='ko', text=content)
        with open(f"../translated_txt/{file_list[i]}_translated.txt", 'a') as f:
            f.write(f"--- {sec} ---")
            f.write("\n")
            f.write(translated_text['translatedText'])
            f.write("\n")