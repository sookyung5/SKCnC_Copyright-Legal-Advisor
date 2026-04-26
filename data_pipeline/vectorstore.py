# -*- coding: utf-8 -*-
"""
벡터스토어 생성 모듈
FAISS 벡터스토어 생성 및 관리
"""
import shutil
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
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


def create_vectorstore_in_batches(docs: List[Document], batch_size: int = 500) -> FAISS:

    CHECKPOINT_DIR = Path(settings.paths.vectorstore_path).parent / "checkpoints"

    if not docs:
        log.error("문서가 비어있습니다")
        raise ValueError("문서 리스트가 비어있습니다")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_model = _get_embeddings_model()

    log.info(f"벡터스토어 생성 시작: 총 {len(docs)}개 문서")
    log.info(f"배치 크기: {batch_size}개")

    # 체크포인트 재개 확인
    checkpoint_path = CHECKPOINT_DIR / "latest"
    start_idx = 0
    vectorstore = None

    if checkpoint_path.exists():
        try:
            log.info("기존 체크포인트 발견 — 이어서 시작")
            vectorstore = FAISS.load_local(
                str(checkpoint_path),
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            start_idx = vectorstore.index.ntotal
            log.info(f"체크포인트 로드 완료: {start_idx}개 벡터 복원")
        except Exception as e:
            log.warning(f"체크포인트 로드 실패, 처음부터 시작: {str(e)}")
            start_idx = 0
            vectorstore = None

    # 배치 처리
    batches = list(range(start_idx, len(docs), batch_size))
    for batch_idx in tqdm(batches, desc="벡터스토어 배치 추가"):
        batch = docs[batch_idx:batch_idx + batch_size]

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    batch, embeddings_model,
                    distance_strategy=DistanceStrategy.COSINE
                )
            else:
                vectorstore.add_documents(batch)

            # 배치마다 체크포인트 저장
            vectorstore.save_local(str(checkpoint_path))
            log.debug(f"진행률: {min(batch_idx + batch_size, len(docs))}/{len(docs)}")

        except Exception as e:
            log.exception(f"배치 오류 (idx {batch_idx}): {str(e)}")
            log.warning("체크포인트까지의 진행상황은 보존됩니다. 재실행 시 이어서 진행됩니다.")
            raise

    # 완료 후 체크포인트 정리
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    log.info("벡터스토어 생성 완료")
    return vectorstore

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
        log.exception(
            f"벡터스토어 저장 오류: {str(e)} (save_path: {save_path})")
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
            embeddings_model
        )
        
        log.info(f"벡터스토어 로드 완료: {load_path}")
        log.info(f"벡터 수: {vectorstore.index.ntotal}")
    
        return vectorstore
    
    except FileNotFoundError:
        raise
    except Exception as e:
        log.exception(
            f"벡터스토어 로드 오류: {str(e)} (load_path: {load_path})")
        raise