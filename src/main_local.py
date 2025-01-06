import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# 기존 deep_translator import 제거
# from deep_translator import GoogleTranslator

def summarize_text(text: str, max_length: int=1000, min_length: int=100) -> str:
    """
    주어진 텍스트를 요약합니다.
    
    Args:
        text (str): 요약할 텍스트
        max_length (int): 요약된 텍스트의 최대 길이
        min_length (int): 요약된 텍스트의 최소 길이
    
    Returns:
        str: 요약된 텍스트
    """
    try:
        summarizer = pipeline("summarization", model="google-t5/t5-small")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return "[요약 실패]"

def translate_text(text: str, source_language: str='en', target_language: str='ko') -> str:
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
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        
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

def extract_text_from_pdf(pdf_path, poppler_path=None):
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
            print(f"Processing page {i + 1}/{len(images)} of {os.path.basename(pdf_path)}...")
            # OCR을 사용하여 이미지에서 텍스트 추출
            page_text = pytesseract.image_to_string(image, lang='eng')  # 언어 설정 가능 (예: 'kor' 한국어)
            text += f"--- Page {i + 1} ---\n{page_text}\n"

        return text

    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({pdf_path}): {e}")
        return ""

def pdf_to_translated_text(pdf_path: str, output_txt_path: str, target_language: str='ko', poppler_path: str=None, 
                           source_language_tesseract: str='eng', target_language_translator: str='ko', 
                           source_language_translator: str='en') -> None:
    """
    PDF 파일을 텍스트로 변환하고, 요약 및 번역하여 저장합니다.
    
    Args:
        pdf_path (str): PDF 파일 경로
        output_txt_path (str): 번역된 텍스트를 저장할 파일 경로
        target_language (str): 번역할 대상 언어 코드
        poppler_path (str, optional): Poppler가 설치된 경로
        source_language_tesseract (str): OCR에서 사용할 언어 코드
        target_language_translator (str): 번역기에서 사용할 대상 언어 코드
        source_language_translator (str): 번역기에서 사용할 원본 언어 코드
    """
    try:
        # PDF에서 텍스트 추출
        extracted_text = pdf_to_text(pdf_path, poppler_path)
        if not extracted_text:
            print(f"텍스트 추출 실패: {pdf_path}")
            return

        # 텍스트 정리
        cleaned_text = clean_text(extracted_text)

        # 섹션 추출
        section_data = extract_sections(cleaned_text)

        if not section_data:
            print(f"섹션 추출 실패: {pdf_path}")
            return

        # 섹션 요약 및 번역
        translated_sections = translate_sections(section_data, target_language=target_language_translator, source_language=source_language_translator)

        # 번역된 섹션 저장
        try:
            if not os.path.exists(os.path.dirname(output_txt_path)):
                os.makedirs(os.path.dirname(output_txt_path))
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                for sec, content in translated_sections.items():
                    f.write(f"--- {sec} ---\n")
                    f.write(content + "\n\n")
            print(f"번역 완료: {output_txt_path}")
        except Exception as e:
            print(f"번역 저장 중 오류 발생 ({pdf_path}): {e}")

    except Exception as e:
        print(f"오류 발생: {e}")

def clean_text(text: str) -> str:
    """
    추출된 텍스트를 정리합니다.
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 정리된 텍스트
    """
    # 정규 표현식을 사용하여 텍스트 정리
    replaced_text = re.sub(r'Page\s*\d+|\s*\d+\s*\n', '', text)
    replaced_text = re.sub(r'-\n', '', replaced_text)
    replaced_text = re.sub(r'\s+', ' ', replaced_text).strip()
    return replaced_text

def extract_sections(text: str) -> dict:
    """
    텍스트에서 섹션을 추출합니다.
    
    Args:
        text (str): 전체 텍스트
    
    Returns:
        dict: 섹션별 텍스트 딕셔너리
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

def translate_sections(section_data: dict, target_language: str='ko', source_language: str='en') -> dict:
    """
    섹션별 텍스트를 요약하고 번역합니다.
    
    Args:
        section_data (dict): 섹션별 텍스트 딕셔너리
        target_language (str): 번역할 대상 언어 코드
        source_language (str): 번역할 원본 언어 코드
    
    Returns:
        dict: 번역된 섹션별 텍스트 딕셔너리
    """
    translated_sections = {}
    for sec, content in section_data.items():
        # 텍스트 요약
        # summary = summarize_text(content)
        
        # 텍스트 번역
        translation = translate_text(text=content, source_language=source_language, target_language=target_language)
        
        translated_sections[sec] = translation

    return translated_sections

def process_pdf(file_name: str, dir_abs_path: str, txt_dir: str, translated_txt_dir: str, poppler_path: str=None, 
               target_language: str='ko', source_language_tesseract: str='eng', 
               target_language_translator: str='ko', source_language_translator: str='en') -> None:
    """
    개별 PDF 파일을 처리합니다: 텍스트 추출, 정리, 섹션 추출, 요약, 번역, 저장.
    
    Args:
        file_name (str): PDF 파일 이름
        dir_abs_path (str): PDF 파일들이 있는 디렉토리 경로
        txt_dir (str): 추출된 텍스트를 저장할 디렉토리
        translated_txt_dir (str): 번역된 텍스트를 저장할 디렉토리
        poppler_path (str, optional): Poppler가 설치된 경로 (필요 시)
        target_language (str): 번역할 대상 언어 코드
        source_language_tesseract (str): OCR에서 사용할 언어 코드
        target_language_translator (str): Translator에서 사용할 대상 언어 코드
        source_language_translator (str): Translator에서 사용할 원본 언어 코드
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
    translated_sections = translate_sections(section_data, target_language=target_language_translator, source_language=source_language_translator)

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
    dir_abs_path = os.path.abspath("../pdf")
    print(dir_abs_path)

    file_lists = os.listdir(dir_abs_path)
    print(file_lists)

    file_list = [f for f in file_lists if os.path.splitext(f)[1].lower() == ".pdf"]
    print(len(file_list))
    
    txt_dir = "../txt"
    translated_txt_dir = "../translated_txt"
    
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.exists(translated_txt_dir):
        os.makedirs(translated_txt_dir)

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU 코어 수: {cpu_count}")

    max_workers = 2
    print(f"최대 작업 수: {max_workers}")

    print("Phase 1: Parallel text extraction")
    with ProcessPoolExecutor(max_workers=2) as executor:  # max_workers는 시스템 자원에 따라 조정
        future_to_file = {executor.submit(extract_text_from_pdf, os.path.join(dir_abs_path, file_name)): file_name for file_name in file_list}
        extracted_texts = {}
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                text = future.result()
                if text:
                    extracted_texts[file_name] = text
                    # Save extracted text to txt_dir
                    output_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"텍스트 추출 완료: {file_name}")
                else:
                    print(f"텍스트 추출 실패: {file_name}")
            except Exception as e:
                print(f"{file_name} 처리 중 오류 발생: {e}")
                
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