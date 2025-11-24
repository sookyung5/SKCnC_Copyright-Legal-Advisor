# -*- coding: utf-8 -*-
"""
청킹 모듈
판례 및 법령 문서 청킹
"""
from curses import meta
from multiprocessing import Value
import re 
import pandas as pd
from typing import Any, List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import log
from .preprocessor import is_nan_value

# ==================== 유틸리티 함수 ====================

def clean_text(text: str) -> str:
    """텍스트 정리"""
    if is_nan_value(text):
        return ""
    return str(text).strip()

def clean_chunk_boundaries(chunks: List[str]) -> List[str]:
    """
    앞쪽의 불필요한 마침표/공백/줄바꿈 제거 함수
    """
    cleaned_chunks = []

    for chunk in chunks:
        # 시작 부분에 붙은 마침표/공백/줄바꿈 제거
        chunk = chunk.lstrip(". \n")

        if chunk:  # 빈 청크 제외
            cleaned_chunks.append(chunk)

    return cleaned_chunks


def natural_sort_key(s:str) -> tuple:
    """
    자연스러운 정렬을 위한 키 함수
    "1" < "1의1" < "1의2" < "2" < "2의1" < "2의2" < "3" < "10" 등
    """
    s = str(s).strip()
    if '의' in s:
        parts = s.split('의')
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0,0)
    else:
        try:
            return (int(s), 0)
        except ValueError:
            return (0,0)


def extract_unique_sorted(items: List) -> List[str]:
    """
    중복 제거 및 자연스러운 정렬
    """
    if not items:
        return []
    
    # 문자열로 변환 및 공백 제거, NaN 값 제외
    str_items = [str(item).strip() for item in items if not is_nan_value(item)]

    if not str_items:
        return []
    
    # set으로 중복 제거
    unique_items = set(str_items)

    # 정렬
    sorted_items = sorted(unique_items, key=natural_sort_key)

    return sorted_items


def validate_chunk_quality(
    chunk: str, 
    min_length: int = None,
) -> Tuple[bool, str]:
    """
    청크 품질 검증
    
    Args:
        chunk: 청크 텍스트
        min_length: 최소 길이        
    Returns:
        (is_valid, error_message)
    """
    min_length = min_length or settings.chunking.min_chunk_size

    if len(chunk) < min_length:
        return False, f"너무 짧음 ({len(chunk)}자)"
    
    
    # 의미 있는 내용 체크
    word_count = len(chunk.split())
    if word_count < 5:
        return False, "내용 부족"
    
    # 섹션 태그 중복 체크
    tags = ['【판시사항】', '【판결요지】', '【판례내용】']
    tag_count = sum(chunk.count(tag) for tag in tags)
    if tag_count > 1:
        return False, "섹션 태그 중복"
    
    return True, "OK"


def chunk_case_documents(df: pd.DataFrame) -> List[Document]:
    """
    판례 문서 청킹
    
    Args:
        df: 판례 데이터프레임
        
    Returns:
        Document 리스트
    """
    log.info(f"판례 청킹 시작: {len(df)}개")
    
    # 필수 컬럼 검증
    required_columns = ['판시사항', '판결요지', '판례내용']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        log.warning(f"누락된 컬럼: {missing_columns}")
    

    text_splitter = RecursiveCharacterTextSplitter(
        separators=settings.chunking.case_separators,
        chunk_size=settings.chunking.case_chunk_size,
        chunk_overlap=settings.chunking.case_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )
    
    case_docs = []
    invalid_chunks = []
    
    
    for idx, row in enumerate(tqdm(
        df.itertuples(index=False), 
        total=len(df), 
        desc="판례 청킹 중"
    )):
        doc_id = f"case_{idx+1}"
        content_parts = []
        
        # 판시사항 추가
        판시사항 = getattr(row, "판시사항", None)
        if not is_nan_value(판시사항):
            content_parts.append(f"【판시사항】\n{clean_text(판시사항)}")
        
        # 판결요지 추가
        판결요지 = getattr(row, "판결요지", None)
        if not is_nan_value(판결요지):
            content_parts.append(f"【판결요지】\n{clean_text(판결요지)}")
        
        # 판례내용 추가
        판례내용 = getattr(row, "판례내용", None)
        if content_parts and not is_nan_value(판례내용):
            content_parts.append(f"【판례내용】\n{clean_text(판례내용)}")
        
        # 내용 결합
        combined_content = "\n\n".join(content_parts)
        
        if not combined_content.strip():
            log.debug(f"{doc_id}: 내용 없음, 건너뜀")
            continue
        
        # 청킹
        try:
            chunks = text_splitter.split_text(combined_content)
        # 빈 청크 제거
            chunks = [c.strip() for c in chunks if c.strip()]
        except Exception as e:
            log.error(f"{doc_id} 청킹 오류: {str(e)}", exc_info=True)
            continue
        
        # 메타데이터 구성
        metadata = {
            "doc_id": doc_id,
            "doc_type": "case",
            "문서유형": "판례"
        }
        
        # 기타 메타데이터 추가
        for key in ["사건명", "사건번호", "선고일자", "법원명", 
                    "사건종류명", "판결유형", "참조조문"]:
            value = getattr(row, key, None)
            if not is_nan_value(value):
                metadata[key] = clean_text(value)
        
        # 각 청크에 대해 Document 생성
        for i, chunk in enumerate(chunks):
            # 청크 품질 검증
            is_valid, error_msg = validate_chunk_quality(
                chunk, 
                min_length=settings.chunking.case_min_chunk_size
            )
            
            if not is_valid:
                invalid_chunks.append((doc_id, i, error_msg))
                continue
            
            # 청크 메타데이터
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "청크번호": i + 1,
                "총청크수": len(chunks)
            })
            
            # Document 생성
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            case_docs.append(doc)
    
    log.info(f"판례 청킹 완료: {len(case_docs)}개 청크 생성")

    # invalid chunk 로깅
    if invalid_chunks:
        log.warning(f"유효하지 않은 청크: {len(invalid_chunks)}개")
    
        # 오류 유형별 통계
        error_stats: Dict[str,int] = {}
        for doc_id, chunk_idx, error_msg in invalid_chunks:
            error_stats[error_msg] = error_stats.get(error_msg, 0) + 1
        
        log.info(f"오류 유형 통계: {error_stats}")

    return case_docs

# 법령 청킹
def chunk_law_documents(df: pd.DataFrame) -> List[Document]:
    """
    법령 문서 청킹 (조문 단위 통합)

    방법:
        1. 조문별로 모든 행 통합 (전문 제외)
        2. chunk_size 이하: 그대로 유지
        3. chunk_size 초과: 청킹
    
    Args:
        df: 법령 데이터프레임
    
    Returns:
        Document 리스트
    """
    log.info(f"법령 청킹 시작: {len(df)}개 행")

    # 조문 단위 문서 생성
    log.info("[1/2] 조문 단위 문서 생성중...")
    article_docs = _create_law_article_documents(df)
    log.info(f"조문 단위 문서 생성 완료: {len(article_docs)}개 조문")

    # 적응형 청킹
    log.info("[2/2] 적응형 청킹 중...")
    final_docs = _adaptive_chunking_law(article_docs)
    log.info(f"적응형 청킹 완료: {len(final_docs)}개 문서")

    log.info(f"법령 청킹 최종 완료: {len(final_docs)}개 문서")

    return final_docs

def _create_law_article_documents(df: pd.DataFrame) -> List[Document]:
    """법령 조문 단위 문서 생성"""
    article_groups = defaultdict(list)

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="조문별 그룹화"
    ):

        # 전문 제외
        article_type = clean_text(getattr(row, '조문여부', ''))
        if article_type == '전문':
            continue

        article_num = clean_text(getattr(row, '조문번호', ''))
        if not article_num:
            continue

        article_title = clean_text(getattr(row, '조문제목', ''))
        key = (article_num, article_title)
        article_groups[key].append(row)

    # 조문 단위 문서 생성
    article_docs = []

    for (article_num, article_title), rows in tqdm(
        article_groups.items(),
        desc="조문 문서 생성"
    ):

        content_parts = []

        # 조문 헤더
        if article_title:
            content_parts.append(f"제{article_num}조 ({article_title})")
        else:
            content_parts.append(f"제{article_num}조")
        
        # 항/호 정보 수집 
        included_hangs = set()
        included_hos = set()

        # 각 행의 내용 추가
        for row in rows:
            row_parts = []

            # 항
            hang_num = clean_text(getattr(row, '항번호', ''))
            hang_content = clean_text(getattr(row, '항내용', ''))
            if hang_num:
                included_hangs.add(hang_num)
            
            if hang_content:
                if hang_num:
                    row_parts.append(f"제{hang_num}항 {hang_content}")
                else:
                    row_parts.append(hang_content)
            
            # 호
            ho_num = clean_text(getattr(row, '호번호', ''))
            ho_content = clean_text(getattr(row, '호내용', ''))
            if ho_num:
                included_hos.append(ho_num)

            if ho_content:
                if ho_num:
                    if re.match(r'^\d+의\d+$', ho_num):
                        row_parts.append(f"  {ho_num} {ho_content}")
                    else:
                        row_parts.append(f"  {ho_num}호 {ho_content}")

                else:
                    row_parts.append(f"  {ho_content}")
            
            # 목
            mok_num = clean_text(getattr(row, '목번호', ''))
            mok_content = clean_text(getattr(row, '목내용', ''))
            if mok_content:
                if mok_num:
                    row_parts.append(f"    {mok_num}목 {mok_content}")
                else:
                    row_parts.append(f"    {mok_content}")

            if row_parts:
                content_parts.append('\n'.join(row_parts))

        content = '\n'.join(content_parts)

        if not content.strip():
            continue
            
        metadata = {
            "doc_type": "law",
            "문서유형": "법령",
            "조문번호": article_num
        }

        if article_title:
            metadata["조문제목"] = article_title
        
        if included_hangs:
            metadata["포함항"] = extract_unique_sorted(list(included_hangs))
        if included_hos:
            metadata["포함호"] = extract_unique_sorted(list(included_hos))
        
        doc = Document(page_content=content.strip(), metadata=metadata)
        article_docs.append(doc)
    
    return article_docs

def _adaptive_chunking_law(article_docs: List[Document]) -> List[Document]:
    """법령 적응형 청킹"""
    splitter = RecursiveCharacterTextSplitter(
        separators=settings.chunking.law_separators,
        chunk_size=settings.chunking.law_chunk_size,
        chunk_overlap=settings.chunking.law_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )
    
    final_docs = []
    chunk_threshold = settings.chunking.law_chunk_size
    invalid_chunks = []
    
    small_count = 0
    chunked_count = 0
    
    for doc in tqdm(article_docs, desc="적응형 청킹"):
        if len(doc.page_content) <= chunk_threshold:
            doc.metadata['doc_id'] = f"law_{len(final_docs)+1}"
            final_docs.append(doc)
            small_count += 1
        else:
            try:
                chunks = splitter.create_documents(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata]
                )
                
                article_num = doc.metadata.get('조문번호', '')
                article_title = doc.metadata.get('조문제목', '')
                
                for i, chunk in enumerate(chunks):
                    is_valid, error_msg = validate_chunk_quality(
                        chunk.page_content,
                        min_length=settings.chunking.law_min_chunk_size
                    )
                    
                    if not is_valid:
                        invalid_chunks.append((article_num, i, error_msg))
                        continue
                    
                    chunk.metadata['doc_id'] = f"law_{len(final_docs)+1}"
                    chunk.metadata['청크번호'] = i + 1
                    chunk.metadata['총청크수'] = len(chunks)
                    
                    if i > 0:
                        if article_title:
                            prefix = f"제{article_num}조 ({article_title}) (계속)\n\n"
                        else:
                            prefix = f"제{article_num}조 (계속)\n\n"
                        chunk.page_content = prefix + chunk.page_content
                    
                    final_docs.append(chunk)
                
                chunked_count += 1
                
            except Exception as e:
                log.error(
                    f"청킹 오류 (조문 {article_num}): {str(e)}",
                    exc_info=True
                )
                doc.metadata['doc_id'] = f"law_{len(final_docs)+1}"
                final_docs.append(doc)
    
    log.info(
        f"적응형 청킹 통계 - "
        f"작은 문서: {small_count}개, "
        f"청킹된 문서: {chunked_count}개, "
        f"최종: {len(final_docs)}개"
    )
    
    if invalid_chunks:
        log.warning(f"유효하지 않은 청크: {len(invalid_chunks)}개")
    
    return final_docs

# 부칙 청킹
def chunk_addendum_documents(df: pd.DataFrame) -> List[Document]:
    """
    부칙 문서 청킹 (조문 단위 통합 + 적응형)
    
    방법:
        1. 부칙 조문별로 모든 행 통합
        2. chunk_size 이하: 그대로 유지
        3. chunk_size 초과: 청킹 적용
    
    Args:
        df: 부칙 데이터프레임
        
    Returns:
        Document 리스트
    """
    log.info(f"부칙 청킹 시작: {len(df)}개 행")
    
    # 1단계: 부칙 조문 단위 문서 생성
    log.info("[1/2] 부칙 조문 단위 문서 생성 중...")
    article_docs = _create_addendum_article_documents(df)
    log.info(f"부칙 조문 단위 문서 생성 완료: {len(article_docs)}개")
    
    # 2단계: 적응형 청킹
    log.info("[2/2] 부칙 적응형 청킹 중...")
    final_docs = _adaptive_chunking_addendum(article_docs)
    log.info(f"부칙 적응형 청킹 완료: {len(final_docs)}개 문서")
    
    log.info(f"부칙 청킹 최종 완료: {len(final_docs)}개 문서")
    return final_docs


def _create_addendum_article_documents(df: pd.DataFrame) -> List[Document]:
    """부칙 조문 단위 문서 생성"""
    article_groups = defaultdict(list)
    
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="부칙 조문별 그룹화"
    ):
        # 기본 정보
        law_no = clean_text(getattr(row, '공포번호', ''))
        law_name = clean_text(getattr(row, '법률제목', ''))
        article_num = clean_text(getattr(row, '부칙_조문번호', ''))
        
        if not article_num:
            continue
        
        article_title = clean_text(getattr(row, '부칙_조문제목', ''))
        
        # 키: (공포번호, 법률제목, 부칙_조문번호, 부칙_조문제목)
        key = (law_no, law_name, article_num, article_title)
        article_groups[key].append(row)
    
    # 부칙 조문 단위 문서 생성
    article_docs = []
    
    for (law_no, law_name, article_num, article_title), rows in tqdm(
        article_groups.items(),
        desc="부칙 문서 생성"
    ):
        content_parts = []
        
        # 부칙 헤더
        if law_name and law_no:
            header = f"법률 제{law_no}호 {law_name} 부칙"
        elif law_no:
            header = f"법률 제{law_no}호 부칙"
        else:
            header = "부칙"
        
        if article_title:
            content_parts.append(f"{header} 제{article_num}조 ({article_title})")
        else:
            content_parts.append(f"{header} 제{article_num}조")
        
        # 부칙내용 (첫 행에서)
        article_content = clean_text(getattr(rows[0], '부칙내용', ''))
        if article_content:
            content_parts.append(article_content)
        
        # 항/호 정보 수집
        included_hangs = []
        included_hos = []

        # 각 행의 항/호 내용 추가
        for row in rows:
            row_parts = []
            
            # 항
            hang_num = clean_text(getattr(row, '항번호', ''))
            hang_content = clean_text(getattr(row, '항내용', ''))
            if hang_num:
                included_hangs.append(hang_num)
            
            if hang_content:
                if hang_num:
                    row_parts.append(f"  제{hang_num}항 {hang_content}")
                else:
                    row_parts.append(f"  {hang_content}")
            
            # 호
            ho_num = clean_text(getattr(row, '호번호', ''))
            ho_content = clean_text(getattr(row, '호내용', ''))
            if ho_num:
                included_hos.append(ho_num)
            
            if ho_content:
                if ho_num:
                    row_parts.append(f"    {ho_num}호 {ho_content}")
                else:
                    row_parts.append(f"    {ho_content}")
            
            if row_parts:
                content_parts.append('\n'.join(row_parts))
        
        content = '\n'.join(content_parts)
        
        if not content.strip():
            continue
        
        metadata = {
            "doc_type": "addendum",
            "문서유형": "부칙",
            "부칙_조문번호": article_num
        }
        
        if law_no:
            metadata["공포번호"] = law_no
        if law_name:
            metadata["법률제목"] = law_name
        if article_title:
            metadata["부칙_조문제목"] = article_title
        
        if included_hangs:
            metadata["포함항"] = extract_unique_sorted(included_hangs)
        if included_hos:
            metadata["포함호"] = extract_unique_sorted(included_hos)

        doc = Document(page_content=content.strip(), metadata=metadata)
        article_docs.append(doc)
    
    return article_docs


def _adaptive_chunking_addendum(article_docs: List[Document]) -> List[Document]:
    """부칙 적응형 청킹"""
    splitter = RecursiveCharacterTextSplitter(
        separators=settings.chunking.addendum_separators,
        chunk_size=settings.chunking.addendum_chunk_size,
        chunk_overlap=settings.chunking.addendum_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )
    
    final_docs = []
    chunk_threshold = settings.chunking.addendum_chunk_size
    invalid_chunks = []
    
    small_count = 0
    chunked_count = 0
    
    for doc in tqdm(article_docs, desc="부칙 적응형 청킹"):
        if len(doc.page_content) <= chunk_threshold:
            doc.metadata['doc_id'] = f"addendum_{len(final_docs)+1}"
            final_docs.append(doc)
            small_count += 1
        else:
            try:
                chunks = splitter.create_documents(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata]
                )
                
                article_num = doc.metadata.get('부칙_조문번호', '')
                article_title = doc.metadata.get('부칙_조문제목', '')
                
                for i, chunk in enumerate(chunks):
                    is_valid, error_msg = validate_chunk_quality(
                        chunk.page_content,
                        min_length=settings.chunking.addendum_min_chunk_size
                    )
                    
                    if not is_valid:
                        invalid_chunks.append((article_num, i, error_msg))
                        continue
                    
                    chunk.metadata['doc_id'] = f"addendum_{len(final_docs)+1}"
                    chunk.metadata['청크번호'] = i + 1
                    chunk.metadata['총청크수'] = len(chunks)
                    
                    if i > 0:
                        if article_title:
                            prefix = f"부칙 제{article_num}조 ({article_title}) (계속)\n\n"
                        else:
                            prefix = f"부칙 제{article_num}조 (계속)\n\n"
                        chunk.page_content = prefix + chunk.page_content
                    
                    final_docs.append(chunk)
                
                chunked_count += 1
                
            except Exception as e:
                log.error(
                    f"부칙 청킹 오류 (조문 {article_num}): {str(e)}",
                    exc_info=True
                )
                doc.metadata['doc_id'] = f"addendum_{len(final_docs)+1}"
                final_docs.append(doc)
    
    log.info(
        f"부칙 적응형 청킹 통계 - "
        f"작은 문서: {small_count}개, "
        f"청킹된 문서: {chunked_count}개, "
        f"최종: {len(final_docs)}개"
    )
    
    if invalid_chunks:
        log.warning(f"유효하지 않은 부칙 청크: {len(invalid_chunks)}개")
    
    return final_docs

# 시행령 청킹
def chunk_enforcement_documents(df: pd.DataFrame) -> List[Document]:
    """
    시행령 문서 청킹 (조문 단위 통합 + 적응형)
    
    방법:
        1. 시행령 조문별로 모든 행 통합
        2. chunk_size 이하: 그대로 유지
        3. chunk_size 초과: 청킹 적용
    
    Args:
        df: 시행령 데이터프레임
        
    Returns:
        Document 리스트
    """
    log.info(f"시행령 청킹 시작: {len(df)}개 행")
    
    # 1단계: 시행령 조문 단위 문서 생성
    log.info("[1/2] 시행령 조문 단위 문서 생성 중...")
    article_docs = _create_enforcement_article_documents(df)
    log.info(f"시행령 조문 단위 문서 생성 완료: {len(article_docs)}개")
    
    # 2단계: 적응형 청킹
    log.info("[2/2] 시행령 적응형 청킹 중...")
    final_docs = _adaptive_chunking_enforcement(article_docs)
    log.info(f"시행령 적응형 청킹 완료: {len(final_docs)}개 문서")
    
    log.info(f"시행령 청킹 최종 완료: {len(final_docs)}개 문서")
    return final_docs


def _create_enforcement_article_documents(df: pd.DataFrame) -> List[Document]:
    """시행령 조문 단위 문서 생성"""
    article_groups = defaultdict(list)
    
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="시행령 조문별 그룹화"
    ):
        article_num = clean_text(getattr(row, '시행령_조문번호', ''))
        
        if not article_num:
            continue
        
        article_title = clean_text(getattr(row, '시행령_조문제목', ''))
        key = (article_num, article_title)
        article_groups[key].append(row)
    
    # 시행령 조문 단위 문서 생성
    article_docs = []
    
    for (article_num, article_title), rows in tqdm(
        article_groups.items(),
        desc="시행령 문서 생성"
    ):
        content_parts = []
        
        # 시행령 헤더
        if article_title:
            content_parts.append(f"제{article_num}조 ({article_title})")
        else:
            content_parts.append(f"제{article_num}조")
        
        # 시행령_조문내용 (첫 행에서)
        article_content = clean_text(getattr(rows[0], '시행령_조문내용', ''))
        if article_content:
            content_parts.append(article_content)
        
        # 항/호 정보 수집
        included_hangs = []
        included_hos = []

        # 각 행의 항/호/목 내용 추가
        for row in rows:
            row_parts = []
            
            # 항
            hang_num = clean_text(getattr(row, '항번호', ''))
            hang_content = clean_text(getattr(row, '항내용', ''))
            if hang_num:
                included_hangs.append(hang_num)
            
            if hang_content:
                if hang_num:
                    row_parts.append(f"  제{hang_num}항 {hang_content}")
                else:
                    row_parts.append(f"  {hang_content}")
            
            # 호
            ho_num = clean_text(getattr(row, '호번호', ''))
            ho_content = clean_text(getattr(row, '호내용', ''))
            if ho_num:
                included_hos.append(ho_num)

            if ho_content:
                if ho_num:
                    row_parts.append(f"    {ho_num}호 {ho_content}")
                else:
                    row_parts.append(f"    {ho_content}")
            
            # 목
            mok_num = clean_text(getattr(row, '목번호', ''))
            mok_content = clean_text(getattr(row, '목내용', ''))
            if mok_content:
                if mok_num:
                    row_parts.append(f"      {mok_num}목 {mok_content}")
                else:
                    row_parts.append(f"      {mok_content}")
            
            if row_parts:
                content_parts.append('\n'.join(row_parts))
        
        content = '\n'.join(content_parts)
        
        if not content.strip():
            continue
        
        metadata = {
            "doc_type": "enforcement_decree",
            "문서유형": "시행령",
            "시행령_조문번호": article_num
        }
        
        if article_title:
            metadata["시행령_조문제목"] = article_title
        if included_hangs:
            metadata["포함항"] = extract_unique_sorted(included_hangs)
        if included_hos:
            metadata["포함호"] = extract_unique_sorted(included_hos)
        
        doc = Document(page_content=content.strip(), metadata=metadata)
        article_docs.append(doc)
    
    return article_docs


def _adaptive_chunking_enforcement(article_docs: List[Document]) -> List[Document]:
    """시행령 적응형 청킹"""
    splitter = RecursiveCharacterTextSplitter(
        separators=settings.chunking.enforcement_separators,
        chunk_size=settings.chunking.enforcement_chunk_size,
        chunk_overlap=settings.chunking.enforcement_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )
    
    final_docs = []
    chunk_threshold = settings.chunking.enforcement_chunk_size
    invalid_chunks = []
    
    small_count = 0
    chunked_count = 0
    
    for doc in tqdm(article_docs, desc="시행령 적응형 청킹"):
        if len(doc.page_content) <= chunk_threshold:
            doc.metadata['doc_id'] = f"enforcement_{len(final_docs)+1}"
            final_docs.append(doc)
            small_count += 1
        else:
            try:
                chunks = splitter.create_documents(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata]
                )
                
                article_num = doc.metadata.get('시행령_조문번호', '')
                article_title = doc.metadata.get('시행령_조문제목', '')
                
                for i, chunk in enumerate(chunks):
                    is_valid, error_msg = validate_chunk_quality(
                        chunk.page_content,
                        min_length=settings.chunking.enforcement_min_chunk_size
                    )
                    
                    if not is_valid:
                        invalid_chunks.append((article_num, i, error_msg))
                        continue
                    
                    chunk.metadata['doc_id'] = f"enforcement_{len(final_docs)+1}"
                    chunk.metadata['청크번호'] = i + 1
                    chunk.metadata['총청크수'] = len(chunks)
                    
                    if i > 0:
                        if article_title:
                            prefix = f"제{article_num}조 ({article_title}) (계속)\n\n"
                        else:
                            prefix = f"제{article_num}조 (계속)\n\n"
                        chunk.page_content = prefix + chunk.page_content
                    
                    final_docs.append(chunk)
                
                chunked_count += 1
                
            except Exception as e:
                log.error(
                    f"시행령 청킹 오류 (조문 {article_num}): {str(e)}",
                    exc_info=True
                )
                doc.metadata['doc_id'] = f"enforcement_{len(final_docs)+1}"
                final_docs.append(doc)
    
    log.info(
        f"시행령 적응형 청킹 통계 - "
        f"작은 문서: {small_count}개, "
        f"청킹된 문서: {chunked_count}개, "
        f"최종: {len(final_docs)}개"
    )
    
    if invalid_chunks:
        log.warning(f"유효하지 않은 시행령 청크: {len(invalid_chunks)}개")
    
    return final_docs