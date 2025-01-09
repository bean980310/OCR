import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import gradio as gr

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

checkpoint='t5-base'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model=T5ForConditionalGeneration.from_pretrained(checkpoint)

def pdf_to_text(pdf_path: str, poppler_path: str=None, lang: str='eng') -> str:
    try:
        # PDF를 이미지로 변환
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        text = ""

        for i, image in enumerate(images):
            print(f"Processing page {i + 1}/{len(images)}...")
            # OCR을 사용하여 이미지에서 텍스트 추출
            page_text = pytesseract.image_to_string(image, lang=lang)  # 언어 설정 가능 (예: 'kor' 한국어)
            text += f"--- Page {i + 1} ---\n{page_text}\n"

        extracted_texts = {}

        file_name = f"{pdf_path}.txt"
            
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return file_name

    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({pdf_path}): {e}")
        return ""
    
    
    
def clean_text(file_name):
    with open(file_name, "r") as f:
        text = f.read()
        
    replaced_text = re.sub(r'Page\s*\d+|\s*\d+\s*\n', '', text)
    replaced_text = re.sub(r'-\n', '', replaced_text)
    replaced_text = re.sub(r'\s+', ' ', replaced_text).strip()
    
    
    if not os.path.exists("../re_txt/"):
        os.makedirs("../re_txt")
     
    file_name=f"{file_name}_cleaned.txt"
    
    with open(file_name, "w") as f:
        f.write(replaced_text)
    
    return file_name

def split_into_sections(file_name):
    with open(file_name, "r") as f:
        replaced_text = f.read()
        
    sections = ["ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHOD", "EXPERIMENT", "CONCLUSION"]
    section_data = {}
    
    for i in range(len(sections)-1):
        pattern = f"{sections[i]}(.*?){sections[i+1]}"
        match = re.search(pattern, replaced_text, re.S | re.I)
        if match:
            section_data[sections[i]] = match.group(1).strip()
    
    last_section = sections[-1]
    pattern = f"{last_section}(.*?)$"
    match = re.search(pattern, replaced_text, re.S | re.I)
    
    if match:
        section_data[last_section] = match.group(1).strip()
    
    new_file_name=f"{file_name}_section.txt"
    file_path=Path.home() / new_file_name
    section=[]
    with open(file_path, 'w') as f:
        f.write("")
    print(f"--- {file_name} ---")

    for sec, content in section_data.items():
        print(f"--- {sec} ---\n{content[:500]}...\n")
        section.append(f"--- {sec} ---\n{content}...\n")
    
    print("-"*25)
    
    for sec, content in section_data.items():
         with open(file_path, 'a') as f:
            f.write(f"--- {sec} ---")
            f.write("\n")
            f.write(content)
            f.write("\n")
            
    return section, file_path
            
def translate_text_m2m(text: str, source_language: str='en', target_language: str='ko') -> str:
    """
    Translates text from source_language to target_language using facebook/m2m100 model.
    
    Args:
        text (str): Text to translate.
        source_language (str): Source language code (ISO 639-1).
        target_language (str): Target language code (ISO 639-1).
        
    Returns:
        str: Translated text.
    """
    try:
        # 모델과 토크나이저 로드
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        
        # 소스 언어 설정
        tokenizer.src_lang = source_language
        
        # 텍스트 토크나이징
        encoded = tokenizer(text, return_tensors="pt")
        
        # 번역 생성
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_language))
        
        # 번역된 토큰 디코딩
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        print(f"Text: {text}")
        print(f"Translation: {translated_text}")
        
        return translated_text
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return "[번역 실패]"

def chunk_text(text, max_tokens, tokenizer):
    """
    텍스트를 최대 토큰 수에 맞게 청킹합니다.
    
    :param text: 원본 텍스트
    :param max_tokens: 한 청크당 최대 토큰 수
    :param tokenizer: Hugging Face 토크나이저
    :return: 청크로 분할된 텍스트 리스트
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length <= max_tokens:
            current_chunk += " " + sentence
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())
        
def filter_token(file_name):
    with open(file_name, "r") as f:
        text = f.read()
    stop_words = set(stopwords.words("english"))
    punctuation = string.punctuation

    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    return filtered_tokens

def split_text(text, chunk_size=512):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length <= chunk_size:
            current_chunk += " " + sentence
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def top_items(file_name):
    with open(file_name, "r") as f:
        text = f.read()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(text)])

    feature_names = vectorizer.get_feature_names_out()
    sorted_items = tfidf_matrix.toarray()[0].argsort()[::-1]
    print("Top TF-IDF words:")
    for idx in sorted_items[:10]:
        return (f"{feature_names[idx]}: {tfidf_matrix[0, idx]}")
        
        
def summarize_text(file_name):
    with open(file_name, "r") as f:
        text = f.read()
    chunks = split_text(text, chunk_size=512)
    summaries = []

    file_name=f"{file_name}_summerized.txt"
    
    for chunk in chunks:
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4)
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        
    final_summary = " ".join(summaries)
        
    print(final_summary)
        
    with open(file_name, 'w') as f:
        f.write(final_summary)
            
    return file_name, final_summary
            
def process_pdf(pdf_file):
    # PDF 파일을 바이트로 읽기
    # OCR을 통해 텍스트 추출
    raw_text = pdf_to_text(pdf_file)
    # 텍스트 정제
    cleaned_text = clean_text(raw_text)

    # 섹션별로 텍스트 분할 (선택 사항)
    sections, split_text = split_into_sections(cleaned_text)

    # 텍스트 요약
    final_file, summary = summarize_text(split_text)

    return final_file, summary

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Summary"),
    title="OCR PDF Summarizer",
    description="Upload a PDF file, and the app will extract text using OCR and provide a summary."
)

# 인터페이스 실행
if __name__ == "__main__":
    iface.launch(inbrowser=True)