# -*- coding: utf-8 -*-
"""
벡터스토어 생성 모듈
FAISS 벡터스토어 생성 및 관리
"""
from typing import List
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from config.settings import settings
from utils.logger import log


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
    """
    log.info(f"벡터스토어 생성 시작: 총 {len(docs)}개 문서")
    log.info(f"배치 크기: {batch_size}개")
    
    # 임베딩 모델 초기화
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai.embedding_model,
        openai_api_key=settings.openai.api_key
    )
    
    # 첫 번째 배치로 초기화
    vectorstore = FAISS.from_documents(
        docs[:batch_size],
        embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    log.info(f"초기 벡터스토어 생성 완료: {batch_size}개 문서")
    
    # 나머지 배치 추가
    for i in range(batch_size, len(docs), batch_size):
        batch_end = min(i + batch_size, len(docs))
        batch = docs[i:batch_end]
        
        batch_vectorstore = FAISS.from_documents(
            batch, 
            embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        
        vectorstore.merge_from(batch_vectorstore)
        log.info(f"진행률: {batch_end}/{len(docs)}")
    
    log.info("벡터스토어 생성 완료")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, save_path: str = None):
    """
    벡터스토어 저장
    
    Args:
        vectorstore: FAISS 벡터스토어
        save_path: 저장 경로 (None이면 설정 파일의 경로 사용)
    """
    if save_path is None:
        save_path = str(settings.paths.vectorstore_path)
    
    vectorstore.save_local(save_path)
    log.info(f"벡터스토어 저장 완료: {save_path}")


def load_vectorstore(load_path: str = None) -> FAISS:
    """
    벡터스토어 로드
    
    Args:
        load_path: 로드 경로 (None이면 설정 파일의 경로 사용)
        
    Returns:
        FAISS 벡터스토어
    """
    if load_path is None:
        load_path = str(settings.paths.vectorstore_path)
    
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai.embedding_model,
        openai_api_key=settings.openai.api_key
    )
    
    vectorstore = FAISS.load_local(
        load_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    
    log.info(f"벡터스토어 로드 완료: {load_path}")
    log.info(f"벡터 수: {vectorstore.index.ntotal}")
    
    return vectorstore
