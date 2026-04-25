# -*- coding: utf-8 -*-
"""
검색 모듈
재시도 로직이 포함된 문서 검색
"""
from typing import Any, List, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import settings
from utils import log
from .reranker import VoyageReranker


class MultiQueryGenerator:
    """멀티쿼리 생성기"""
    
    def __init__(self):
        """초기화"""
        self.llm = ChatOpenAI(
            model_name=settings.openai.evaluation_model,
            temperature=0.7,
            openai_api_key=settings.openai.api_key,
            max_tokens=settings.openai.max_tokens_multiquery,
            request_timeout=30
        )
        
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""원본 질문을 다양한 관점에서 3개의 검색 쿼리로 변환하세요.
각 쿼리는 다른 측면을 강조해야 합니다.

원본 질문: {query}

검색 쿼리 3개를 아래 형식으로 작성:
1. [첫 번째 쿼리]
2. [두 번째 쿼리]
3. [세 번째 쿼리]"""
        )
        
        log.info(f"멀티쿼리 생성기 초기화 완료 (max_tokens={settings.openai.max_tokens_multiquery})")
    
    def generate(self, query: str) -> List[str]:
        """f
        멀티쿼리 생성
        
        Args:
            query: 원본 질문
            
        Returns:
            생성된 쿼리 리스트 (3개)
        """
        try:
            prompt = self.prompt.format(query=query)
            response = self.llm.predict(prompt)
            
            # 줄 단위로 파싱
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 4)):
                    # "1. " 제거
                    query_text = line.split('.', 1)[1].strip()
                    queries.append(query_text)
            
            # 정확히 3개가 아니면 원본 포함
            if len(queries) != 3:
                log.warning(f"멀티쿼리 생성 실패 (생성된 개수: {len(queries)}), 원본 사용")
                return [query]
            
            return queries
        
        except Exception as e:
            log.error(f"멀티쿼리 생성 오류: {str(e)}")
            return [query]  # 실패시 원본만 반환


class LegalRetriever:
    """재시도 로직이 포함된 법률 문서 검색기"""
    
    def __init__(self, vectorstore, use_reranker: bool = True):
        """
        초기화
        
        Args:
            vectorstore: FAISS 벡터스토어
            use_reranker: 리랭커 사용 여부
        """
        self.vectorstore = vectorstore
        self.multi_query_gen = MultiQueryGenerator()
        self.use_reranker = use_reranker
        
        if use_reranker:
            self.reranker = VoyageReranker()
        
        log.info(f"검색기 초기화 완료 (리랭커: {use_reranker})")
    
    def _select_k(self, retry_num: int) -> int:
        """
        retry 단계별 검색 문서 수(k) 선택
        
        Args:
            retry_num: 재시도 번호 (0, 1, 2)
            
        Returns:
            검색할 문서 수
        """
        if retry_num <= 0:
            return settings.retrieval.retry_0_k
        elif retry_num == 1:
            return settings.retrieval.retry_1_k
        else:
            return settings.retrieval.retry_2_k
    
    def _build_queries(self, query: str, retry_num: int) -> List[str]:
        """
        retry 단계별 검색 쿼리 생성
        
        Args:
            query: 원본 질문
            retry_num: 재시도 번호 (0, 1, 2)
            
        Returns:
            검색 쿼리 리스트
        """
        if retry_num == 0:
            log.info("[시도 1] 기본 검색 (단일 쿼리)")
            return [query]
        else:
            log.info(f"[시도 {retry_num + 1}] 멀티쿼리 검색")
            return self.multi_query_gen.generate(query)

    
    def _dedup_docs(self, docs: List[Document]) -> List[Document]:
        """
        문서 중복 제거

        Args:
            docs: 원본 문서 리스트

        Returns:
            중복 제거된 문서 리스트
        """
        seen_content = set()
        unique_docs = []

        for doc in docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        return unique_docs


    def retrieve(
        self, 
        query: str, 
        retry_num: int = 0
    ) -> Tuple[List[Document], List[Document]]:
        """
        문서 검색 + 리랭킹
        
        Args:
            query: 검색 쿼리
            retry_num: 재시도 번호 (0, 1, 2)
            
        Returns:
            (원본 검색 문서, 리랭킹된 문서) 튜플
        """
        try:
            # k값 선택
            k = self._select_k(retry_num)

            # 쿼리 생성
            queries = self._build_queries(query, retry_num)

            log.info(f"검색 시작 (k={k}, queries={len(queries)})")
            
            # 멀티쿼리 검색
            all_docs = []
            for q in queries:
                docs = self.vectorstore.similarity_search(q, k=k)
                all_docs.extend(docs)
            
            # 중복 제거
            unique_docs: List[Document] = self._dedup_docs(all_docs)
            log.info(f"검색 완료: {len(unique_docs)}개 문서 (retry={retry_num})")
            
            # 리랭킹
            if self.use_reranker and unique_docs:
                reranked_docs = self.reranker.rerank(query, unique_docs)
            else:
                reranked_docs = unique_docs[:settings.retrieval.rerank_top_k]
            
            return unique_docs, reranked_docs
        
        except Exception as e:
            log.error(f"검색 오류: {str(e)}")
            return [], []
