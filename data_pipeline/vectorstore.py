# -*- coding: utf-8 -*-
"""
벡터스토어 생성 모듈
FAISS 벡터스토어 생성 및 관리
"""
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from config.settings import settings
from utils.logger import log

def _get_embeddings_model() -> OpenAIEmbeddings:
    """
    임베딩 모델 생성 (재사용을 위한 헬퍼 함수)
    
    Returns:
        OpenAI 임베딩 모델
    """
    return OpenAIEmbeddings(
        model=settings.openai.embedding_model,
        openai_api_key=settings.openai.api_key
    )

def create_vectorstore_in_batches(
    docs: List[Document],
    batch_size: int = 500
) -> FAISS:
    """
    배치 단위로 벡터스토어 생성
    
    Args:
        docs: Document 리스트
        batch_size: 배치 크기
        
    Returns:
        FAISS 벡터스토어
    
    Raises:
        ValueError: 문서가 비어있을 때
        Exception: 벡터스토어 생성 실패
    """

    if not docs:
        log.error("문서가 비어있습니다")
        raise ValueError("문서 리스트가 비어있습니다")

    log.info(f"벡터스토어 생성 시작: 총 {len(docs)}개 문서")
    log.info(f"배치 크기: {batch_size}개")
    
    try:
        # 임베딩 모델 초기화
        embeddings_model = _get_embeddings_model()
        
        # 첫 번째 배치로 초기화
        first_batch_size = min(batch_size, len(docs))
        vectorstore = FAISS.from_documents(
            docs[:first_batch_size],
            embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        log.info(f"초기 벡터스토어 생성 완료: {first_batch_size}개 문서")
        
        # 나머지 배치 추가
        if len(docs) > batch_size:
            total_batches = (len(docs) - batch_size + batch_size - 1) // batch_size

            for batch_idx in tqdm(
                range(batch_size, len(docs), batch_size),
                desc="벡터스토어 배치 추가",
                total=total_batches
            ):
                batch_end = min(batch_idx + batch_size, len(docs))
                batch = docs[batch_idx:batch_end]

                vectorstore.add_documents(batch)
                
                log.debug(f"진행률: {batch_end}/{len(docs)}")

        log.info("벡터스토어 생성 완료")
        return vectorstore
    except Exception as e:
        log.error(
            f"벡터스토어 생성 오류: {str(e)}",
            exc_info=True,
            extra={"doc_count": len(docs)}
        )
        raise

def save_vectorstore(vectorstore: FAISS, save_path: str = None):
    """
    벡터스토어 저장
    
    Args:
        vectorstore: FAISS 벡터스토어
        save_path: 저장 경로 (None이면 설정 파일의 경로 사용)
    
    Raises:
        Exception: 저장 실패 
    """
    try:
        if save_path is None:
            save_path = str(settings.paths.vectorstore_path)
        
        # 디렉토리 생성 
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(save_path)
        log.info(f"벡터스토어 저장 완료: {save_path}")
        log.info(f"벡터 수: {vectorstore.index.ntotal}")
    
    except Exception as e:
        log.error(
            f"벡터스토어 저장 오류: {str(e)}",
            exc_info=True,
            extra={"save_path": save_path}
        )
        raise


def load_vectorstore(load_path: str = None) -> FAISS:
    """
    벡터스토어 로드
    
    Args:
        load_path: 로드 경로 (None이면 설정 파일의 경로 사용)
        
    Returns:
        FAISS 벡터스토어
    
    Raises:
        FileNotFoundError: 벡터스토어 파일이 없을 때
        Exception: 로드 실패
    """
    try:
        if load_path is None:
            load_path = str(settings.paths.vectorstore_path)
        
        # 경로 존재 확인
        if not Path(load_path).exists():
            log.error(f"벡터스토어가 존재하지 않습니다: {load_path}")
            raise FileNotFoundError(f"벡터스토어를 찾을 수 없습니다: {load_path}")

        embeddings_model = _get_embeddings_model()
        
        vectorstore = FAISS.load_local(
            load_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        log.info(f"벡터스토어 로드 완료: {load_path}")
        log.info(f"벡터 수: {vectorstore.index.ntotal}")
    
        return vectorstore
    
    except FileNotFoundError:
        raise
    except Exception as e:
        log.error(
            f"벡터스토어 로드 오류: {str(e)}",
            exc_info=True,
            extra={"load_path": load_path}
        )
        raise