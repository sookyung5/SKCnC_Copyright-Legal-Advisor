# -*- coding: utf-8 -*-
"""
리랭킹 모듈
Voyage Reranker를 사용한 문서 재정렬
"""
from typing import List, Optional
from langchain.schema import Document
import voyageai

from config import settings
from utils import log
from copy import deepcopy

class VoyageReranker:
    """Voyage AI Reranker"""
    
    def __init__(self, api_key: Optional[str] = None, top_k: Optional[int] = None):
        """
        초기화
        
        Args:
            api_key: Voyage API 키 (없으면 설정에서 가져옴)
            top_k: 상위 k개 문서 선택 (없으면 설정에서 가져옴)
        """
        self.api_key = api_key or settings.voyage.api_key
        if not self.api_key:
            raise ValueError(
                "Voyage API 키가 설정되지 않았습니다."
            )

        self.model = settings.voyage.reranker_model
        self.top_k = top_k or settings.voyage.top_k
        self.client = voyageai.Client(api_key=self.api_key)
        
        log.info(f"Voyage Reranker 초기화 완료 (모델: {self.model}, top_k: {self.top_k})")
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        문서 리랭킹
        
        Args:
            query: 검색 쿼리
            documents: 원본 문서 리스트
            
        Returns:
            리랭킹된 문서 리스트 (상위 k개)
        """
        if not documents:
            return []
        
        try:
            log.info(f"리랭킹 시작: {len(documents)}개 문서")
            
            texts = [doc.page_content for doc in documents]
            top_k = min(self.top_k, len(documents))

            result = self.client.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_k=top_k
            )
            
            
            reranked_docs = []
            for item in result.results:
                # 원본은 보존하고 복사본을 수정 
                doc_copy = deepcopy(documents[item.index])
                doc_copy.metadata["rerank_score"] = item.relevance_score
                reranked_docs.append(doc_copy)
            
            log.info(f"리랭킹 완료: {len(reranked_docs)}개 문서 선택")
            return reranked_docs
        
        except Exception as e:
            log.error(f"리랭킹 오류: {str(e)}")
            log.warning("리랭킹 실패 - 원본 문서 반환")
            return documents[:self.top_k]
