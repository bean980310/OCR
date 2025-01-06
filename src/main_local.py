import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from deep_translator import GoogleTranslator
from transformers import pipeline
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Tesseract 실행 파일 경로 설정 (Windows의 경우)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def summarize_text(text, max_length=2000, min_length=500):
    """
    주어진 텍스트를 요약합니다.
    
    :param text: 요약할 텍스트
    :param max_length: 요약된 텍스트의 최대 길이
    :param min_length: 요약된 텍스트의 최소 길이
    :return: 요약된 텍스트
    """
    try:
        summarizer = pipeline("summarization", model="google-t5/t5-base")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return "[요약 실패]"

def translate_text(text, target_language='ko', source_language='en'):
    """
    주어진 텍스트를 번역합니다.
    
    :param text: 번역할 텍스트
    :param target_language: 번역할 대상 언어 코드 (예: 'ko' 한국어)
    :param source_language: 원본 언어 코드 (예: 'en' 영어)
    :return: 번역된 텍스트
    """
    try:
        translator = GoogleTranslator(source=source_language, target=target_language)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return "[번역 실패]"

def pdf_to_text(pdf_path, poppler_path=None):
    """
    PDF 파일에서 텍스트를 추출합니다.
    
    :param pdf_path: PDF 파일 경로
    :param poppler_path: Poppler가 설치된 경로 (필요 시)
    :return: 추출된 텍스트
    """
    try:
        # PDF를 이미지로 변환
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        text = ""

        for i, image in enumerate(images):
            print(f"Processing page {i + 1}/{len(images)}...")
            # OCR을 사용하여 이미지에서 텍스트 추출
            page_text = pytesseract.image_to_string(image, lang='eng')  # 언어 설정 가능 (예: 'kor' 한국어)
            text += f"--- Page {i + 1} ---\n{page_text}\n"

        return text

    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({pdf_path}): {e}")
        return ""

        
def pdf_to_translated_text(pdf_path, output_txt_path, target_language='ko', poppler_path=None, 
                           source_language_tesseract='eng', target_language_translator='en', 
                           source_language_translator='en'):
    try:
        # PDF를 이미지로 변환
        images = convert_from_path(pdf_path, poppler_path=poppler_path)

        translator = GoogleTranslator(source=source_language_translator, target=target_language_translator)
        translated_text = ""

        for i, image in enumerate(images):
            print(f"페이지 {i + 1}/{len(images)} 처리 중...")
            # OCR을 사용하여 이미지에서 텍스트 추출
            page_text = pytesseract.image_to_string(image, lang=source_language_tesseract)  # 언어 설정 (예: 'eng' 영어)

            if not page_text.strip():
                print(f"페이지 {i + 1}에서 텍스트를 추출할 수 없습니다.")
                continue

            # 텍스트 요약
            print("텍스트 요약 중...")
            try:
                summary = summarize_text(page_text)
            except Exception as e:
                print(f"요약 중 오류 발생: {e}")
            finally:
                summary = summarize_text(page_text)

            # 텍스트 번역
            print("텍스트 번역 중...")
            try:
                translation = translator.translate(summary)
            except Exception as e:
                print(f"번역 중 오류 발생: {e}")
                translation = "[번역 실패]"

            translated_text += f"--- 페이지 {i + 1} ---\n{translation}\n"

        # 번역된 텍스트를 파일에 저장
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)

        print(f"번역 완료! 결과는 {output_txt_path}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

        
def clean_text(text):
    """
    추출된 텍스트를 정리합니다.
    
    :param text: 원본 텍스트
    :return: 정리된 텍스트
    """
    # 정규 표현식을 사용하여 텍스트 정리
    replaced_text = re.sub(r'Page\s*\d+|\s*\d+\s*\n', '', text)
    replaced_text = re.sub(r'-\n', '', replaced_text)
    replaced_text = re.sub(r'\s+', ' ', replaced_text).strip()
    return replaced_text


def extract_sections(text):
    """
    텍스트에서 섹션을 추출합니다.
    
    :param text: 전체 텍스트
    :return: 섹션별 텍스트 딕셔너리
    """
    sections = ["ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHOD", "EXPERIMENT", "CONCLUSION"]
    section_data = {}

    for j in range(len(sections)-1):
        pattern = f"{sections[j]}(.*?){sections[j+1]}"
        match = re.search(pattern, text, re.S | re.I)
        if match:
            section_data[sections[j]] = match.group(1).strip()

    last_section = sections[-1]
    pattern = f"{last_section}(.*?)$"
    match = re.search(pattern, text, re.S | re.I)

    if match:
        section_data[last_section] = match.group(1).strip()

    return section_data

def translate_sections(section_data, target_language='ko'):
    """
    섹션별 텍스트를 요약하고 번역합니다.
    
    :param section_data: 섹션별 텍스트 딕셔너리
    :param target_language: 번역할 대상 언어 코드
    :return: 번역된 섹션별 텍스트 딕셔너리
    """
    translated_sections = {}
    for sec, content in section_data.items():
        # 텍스트 요약
        summary = summarize_text(content)
        
        # 텍스트 번역
        translation = translate_text(summary, target_language=target_language)
        
        translated_sections[sec] = translation

    return translated_sections
def process_pdf(file_name, dir_abs_path, txt_dir, translated_txt_dir, poppler_path=None, 
               target_language='ko', source_language_tesseract='eng', 
               target_language_translator='ko', source_language_translator='en'):
    """
    개별 PDF 파일을 처리합니다: 텍스트 추출, 정리, 섹션 추출, 요약, 번역, 저장.
    
    :param file_name: PDF 파일 이름
    :param dir_abs_path: PDF 파일들이 있는 디렉토리 경로
    :param txt_dir: 추출된 텍스트를 저장할 디렉토리
    :param translated_txt_dir: 번역된 텍스트를 저장할 디렉토리
    :param poppler_path: Poppler가 설치된 경로 (필요 시)
    :param target_language: 번역할 대상 언어 코드
    :param source_language_tesseract: OCR에서 사용할 언어 코드
    :param target_language_translator: Translator에서 사용할 대상 언어 코드
    :param source_language_translator: Translator에서 사용할 원본 언어 코드
    """
    pdf_path = os.path.join(dir_abs_path, file_name)
    output_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
    output_translated_path = os.path.join(translated_txt_dir, f"{file_name}_translated.txt")

    # PDF에서 텍스트 추출
    extracted_text = pdf_to_text(pdf_path, poppler_path)
    if not extracted_text:
        print(f"텍스트 추출 실패: {file_name}")
        return

    # 텍스트 정리
    cleaned_text = clean_text(extracted_text)

    # 섹션 추출
    section_data = extract_sections(cleaned_text)

    if not section_data:
        print(f"섹션 추출 실패: {file_name}")
        return

    # 섹션 요약 및 번역
    translated_sections = translate_sections(section_data, target_language=target_language)

    # 번역된 섹션 저장
    try:
        if not os.path.exists(translated_txt_dir):
            os.makedirs(translated_txt_dir)
        with open(output_translated_path, 'w', encoding='utf-8') as f:
            for sec, content in translated_sections.items():
                f.write(f"--- {sec} ---\n")
                f.write(content + "\n\n")
        print(f"번역 완료: {output_translated_path}")
    except Exception as e:
        print(f"번역 저장 중 오류 발생 ({file_name}): {e}")


if __name__ == "__main__":
    dir_abs_path=os.path.abspath("../pdf")
    print(dir_abs_path)

    file_lists=os.listdir(dir_abs_path)
    print(file_lists)

    file_list=[f for f in file_lists if os.path.splitext(f)[1].lower() == ".pdf"]
    print(len(file_list))
    
    txt_dir = "../txt"
    translated_txt_dir = "../translated_txt"
    
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.exists(translated_txt_dir):
        os.makedirs(translated_txt_dir)

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU 코어 수: {cpu_count}")

    max_workers = cpu_count-2

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_name in file_list:
            futures.append(executor.submit(
                process_pdf, 
                file_name, 
                dir_abs_path, 
                txt_dir, 
                translated_txt_dir,
                poppler_path=None,
                target_language='kor',
                source_language_tesseract='eng',
                target_language_translator='ko',
                source_language_translator='en'
            ))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"프로세스 중 오류 발생: {e}")