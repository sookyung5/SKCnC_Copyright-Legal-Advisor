# -*- coding: utf-8 -*-
"""
전처리 모듈
판례 데이터 전처리 (주문 추출, 참조조문 정리 등)
"""
import re
import numpy as np
import pandas as pd
from utils.logger import log


def is_nan_value(value) -> bool:
    """NaN 값 검증"""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.lower() in ['nan', '<na>', 'none']:
        return True
    return False


def extract_judgment_content(text: str) -> str:
    """
    판례 내용에서 주문 부분만 추출
    
    Args:
        text: 전체 판례 텍스트
        
    Returns:
        주문 부분 텍스트 (【주문】 태그부터 끝까지)
    """
    if is_nan_value(text):
        return np.nan
    
    match = re.search(r'【주\s*문】(.*?)$', text, re.DOTALL)
    
    if match:
        return match.group(0).strip()
    else:
        log.warning("【주문】 태그를 찾을 수 없습니다")
        return np.nan


def clean_reference_articles(ref_str: str) -> str:
    """
    참조조문에서 괄호 안의 내용 제거
    
    Args:
        ref_str: 참조조문 문자열
        
    Returns:
        정리된 참조조문
    """
    if is_nan_value(ref_str):
        return ""
    
    # 괄호와 내용 제거
    cleaned = re.sub(r'\([^)]*\)', '', ref_str)
    return cleaned.strip()


def preprocess_case_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    판례 데이터 전처리
    
    Args:
        df: 원본 판례 데이터프레임
        
    Returns:
        전처리된 데이터프레임
    """
    log.info(f"전처리 시작: {len(df)}개 판례")
    
    # 1. 주문 내용 추출
    log.info("주문 내용 추출 중...")
    df['판례내용'] = df['판례내용'].apply(extract_judgment_content)
    
    # 2. 참조조문 정리
    log.info("참조조문 정리 중...")
    df['참조조문'] = df['참조조문'].apply(clean_reference_articles)
    
    log.info("전처리 완료")
    return df
