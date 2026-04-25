# -*- coding: utf-8 -*-
"""
RAG 파이프라인 모듈
전체 프로세스 조율
"""
from typing import List, Optional
from dataclasses import dataclass

from config import settings
from utils import log
from .intent_analyzer import IntentAnalyzer, IntentResult
from .retriever import LegalRetriever
from .generator import AnswerGenerator
from .evaluator import RAGASEvaluator, EvaluationResult
from langchain.schema import Document



@dataclass
class RAGResult:
    """RAG 파이프라인 최종 결과"""
    query: str
    answer: str
    source_documents: List[Document]
    reranked_documents: List[Document]
    evaluation: EvaluationResult
    retry_count: int
    is_fallback_used: bool
    intent: IntentResult


class LegalRAGPipeline:
    """법률 RAG 파이프라인"""
    
    def __init__(
        self, 
        vectorstore, 
        max_retry: Optional[int] = None,    
        use_reranker: Optional[bool] = None
    ):
        """
        초기화
        
        Args:
            vectorstore: FAISS 벡터스토어
            max_retry: 최대 재시도 횟수 (None이면 설정에서 가져옴)
            use_reranker: 리랭커 사용 여부 (None이면 설정에서 가져옴)
        """
        self.max_retry = max_retry or settings.retry.max_retry_count
        use_reranker = use_reranker if use_reranker is not None else settings.retrieval.use_reranker
        
        self.intent_analyzer = IntentAnalyzer()
        self.retriever = LegalRetriever(vectorstore, use_reranker)
        self.generator = AnswerGenerator()
        self.evaluator = RAGASEvaluator(
            faithfulness_threshold=settings.evaluation.min_passing_score,  # 0.5
            relevancy_threshold=settings.evaluation.min_passing_score,  # 0.7
            fallback_faithfulness_min=settings.evaluation.fallback_min_faithfulness, # 0.1
            fallback_faithfulness_max=settings.evaluation.min_passing_score,  # 0.5
            fallback_relevancy_threshold=settings.evaluation.fallback_min_relevancy # 0.8
        )
        
        log.info(f"법률 RAG 파이프라인 초기화 완료 "
                f"(최대 재시도: {self.max_retry}, 리랭커: {use_reranker})")
    
    def process(self, query: str) -> RAGResult:
        """
        질문 처리 및 재시도 로직 
        
        재시도 전략:
            - 1차 시도: 기본 검색 (k=10, 단일 쿼리)
            - 2차 시도: 멀티쿼리 검색 (k=15)
            - 3차 시도: 최대 검색 (k=20, 멀티쿼리)
        
        평가 기준:
            - 통과: faithfulness >= 0.5 AND relevancy >= 0.7
            - Fallback: 0.1 < faithfulness < 0.5 AND relevancy >= 0.8
            - 실패: 그 외 
        
        Args:
            query: 사용자 질문
            
        Returns:
            RAGResult 객체
        
        Raises:
            ValueError: 설정 오류
            Exception: 파이프라인 실행 오류 
        """
        log.info("=" * 80)
        log.info(f"RAG 파이프라인 시작: {query}")
        log.info("=" * 80)
        
        try:
            # 1. 의도 분석
            log.info("\n[단계 1] 의도 분석")
            intent = self.intent_analyzer.analyze(query)
            log.info(f"의도 분석 완료 - 저작권법 관련: {intent.is_copyright_related}, "
                    f"문서 유형: {intent.doc_type}, "
                    f"신뢰도: {intent.confidence:.2f}")
            
            # 저작권법 무관 질문 처리 
            if not intent.is_copyright_related:
                return self._handle_non_copyright_query(query, intent)
            
            # 2. 재시도 루프
            retry_num = 0
            best_fallback = None
            
            while retry_num < self.max_retry:
                log.info(f"\n{'=' * 80}")
                log.info(f"[재시도 {retry_num + 1}/{self.max_retry}]")
                log.info(f"{'=' * 80}")
                
                # 2-1. 문서 검색
                log.info("\n[단계 2] 문서 검색")
                original_docs, reranked_docs = self.retriever.retrieve(
                    query, retry_num
                )
                
                # 검색 결과 없음 
                if not reranked_docs:
                    log.warning("검색된 문서가 없습니다")
                    if settings.retry.retry_on_empty_result:
                        retry_num += 1
                        continue
                    else:
                        break
                
                # 2-2. 답변 생성
                log.info("\n[단계 3] 답변 생성")
                answer = self.generator.generate(query, reranked_docs)
                log.info(f"답변 생성 완료 (길이: {len(answer)}자)")
                
                # 2-3. RAGAS 평가
                log.info("\n[단계 4] RAGAS 평가")
                evaluation = self.evaluator.evaluate(query, answer, reranked_docs)
                log.info(f"평가 결과 - {evaluation.feedback}")
                
                # 2-4. 결과 판정
                if evaluation.passed:
                    # 통과! 
                    log.info(f"\n✅ 통과! (시도 {retry_num + 1}회)")
                    return RAGResult(
                        query=query,
                        answer=answer,
                        source_documents=original_docs,
                        reranked_documents=reranked_docs,
                        evaluation=evaluation,
                        retry_count=retry_num,
                        is_fallback_used=False,
                        intent=intent
                    )
                
                # Fallback 조건 체크
                elif evaluation.is_fallback:
                    # Fallback 후보
                    log.info(f"⚠️ Fallback 대상")
                    if (best_fallback is None or
                        self._is_better_fallback(evaluation, best_fallback[3])):
                        best_fallback = (
                            answer,
                            original_docs,
                            reranked_docs,
                            evaluation,
                            retry_num
                        )
                        log.info(
                            f"현재 최고 Fallback 갱신 "
                            f"(관련성: {evaluation.relevancy_score:.2f})"
                        )
                
                retry_num += 1
            
            # 3. 최종 결과 반환 
            if best_fallback:
                log.warning(f"\n⚠️ {self.max_retry}회 시도 후 통과 실패")
                log.info("최선의 Fallback 결과 반환")
                
                answer, original_docs, reranked_docs, evaluation, retry_num = best_fallback

                log.info(
                    f"선택된 Fallback - 시도 {retry_num + 1}회, "
                    f"충실성: {evaluation.faithfulness_score:.2f}, "
                    f"관련성: {evaluation.relevancy_score:.2f},"
                )

                return RAGResult(
                    query=query,
                    answer=answer,
                    source_documents=original_docs,
                    reranked_documents=reranked_docs,
                    evaluation=evaluation,
                    retry_count=retry_num,
                    is_fallback_used=True,
                    intent=intent
                )
            
            else:
                # 검색 결과 없음 
                return self._handle_no_results(query, intent, retry_num)
        
        except ValueError as e:
            log.error(f"설정 오류: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            log.error(
                f"파이프라인 오류: {str(e)}",
                exc_info=True,
                extra={"query": query} # 컨텍스트 정보 추가 
            )
            raise
    def _is_better_fallback(
        self,
        new_eval: EvaluationResult,
        old_eval: EvaluationResult
    ) -> bool:
        """
        Fallback 점수 비교 (가중치 고려)
        
        Args:
            new_eval: 새 평가 결과
            old_eval: 기존 평가 결과
            
        Returns:
            새 결과가 더 좋으면 True
        """
        # relevancy만 비교 → 가중치 적용한 종합 점수 비교
        # 기존: key=lambda x: x.evaluation.relevancy_score
        # 수정: faithfulness와 relevancy 가중 평균
        new_score = (
            new_eval.faithfulness_score * settings.evaluation.faithfulness_weight +
            new_eval.relevancy_score * settings.evaluation.relevancy_weight
        )
        old_score = (
            old_eval.faithfulness_score * settings.evaluation.faithfulness_weight +
            old_eval.relevancy_score * settings.evaluation.relevancy_weight
        )
        return new_score > old_score

    def _handle_non_copyright_query(
        self, 
        query: str, 
        intent: IntentResult
    ) -> RAGResult:
        """저작권법 무관 질문 처리"""
        answer = """죄송합니다. 이 질문은 저작권법과 관련이 없는 것으로 판단됩니다.

저는 저작권법 전문 AI로, 저작권법 관련 질문에만 답변할 수 있습니다.

다음과 같은 질문에 답변할 수 있습니다:
- 저작권 침해 여부
- 저작재산권과 저작인격권
- 공정이용 판단 기준
- 저작물 이용 허락
- 저작권 등록 및 보호

저작권법 관련 질문이 있으시면 다시 물어봐 주세요."""
        
        evaluation = EvaluationResult(
            faithfulness_score=0.0,
            relevancy_score=0.0,
            passed=False,
            is_fallback=False,
            feedback="저작권법 무관 질문"
        )
        
        log.info("저작권법 무관 질문으로 판단 - 안내 메시지 반환") # 로그 추가 

        return RAGResult(
            query=query,
            answer=answer,
            source_documents=[],
            reranked_documents=[],
            evaluation=evaluation,
            retry_count=0,
            is_fallback_used=False,
            intent=intent
        )
    
    def _handle_no_results(
        self, 
        query: str, 
        intent: IntentResult, 
        retry_count: int
    ) -> RAGResult:
        """검색 결과 없음 처리"""
        answer = """죄송합니다. 관련된 법령이나 판례를 찾을 수 없습니다.

다음 방법을 시도해 보세요:
1. 질문을 더 구체적으로 작성
2. 다른 키워드 사용
3. 질문을 단순화

예시:
- "저작권 침해 기준은?" → "저작권 침해는 어떤 경우에 성립하나요?"
- "30조" → "저작권법 제30조의 내용은 무엇인가요?"

도움이 필요하시면 다시 질문해 주세요."""
        
        evaluation = EvaluationResult(
            faithfulness_score=0.0,
            relevancy_score=0.0,
            passed=False,
            is_fallback=False,
            feedback="검색 결과 없음"
        )
        
        log.warning(f"{retry_count}회 시도 후 검색 결과 없음") # 로그 추가

        return RAGResult(
            query=query,
            answer=answer,
            source_documents=[],
            reranked_documents=[],
            evaluation=evaluation,
            retry_count=retry_count,
            is_fallback_used=False,
            intent=intent
        )


# QA Chain 래퍼 
class LegalQAChain:
    """기존 코드 호환을 위한 QA Chain 래퍼"""
    
    def __init__(self, vectorstore=None):
        """초기화"""
        if vectorstore is None:
            from data_pipeline.vectorstore import load_vectorstore
            vectorstore = load_vectorstore()
        
        self.pipeline = LegalRAGPipeline(vectorstore)
        log.info("LegalQAChain 초기화 완료")
    
    def run(self, query: str) -> dict:
        """
        기존 인터페이스 호환
        
        Args:
            query: 질문
            
        Returns:
            {"query": ..., "answer": ..., "source_documents": ...}
        """
        result = self.pipeline.process(query)
        
        return {
            "query": result.query,
            "answer": result.answer,
            "source_documents": result.reranked_documents
        }
