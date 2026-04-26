# -*- coding: utf-8 -*-
"""
평가 모듈
RAGAS를 사용한 답변 평가
"""
from typing import List
from dataclasses import dataclass
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

from config import settings
from utils import log


@dataclass
class EvaluationResult:
    """평가 결과"""
    faithfulness_score: float
    relevancy_score: float
    passed: bool  # 통과 여부
    is_fallback: bool  # fallback 대상 여부
    feedback: str # 피드백 메시지 


class RAGASEvaluator:
    """RAGAS 평가 클래스"""
    
    def __init__(
        self, 
        faithfulness_threshold: float = None, # 기본값 None으로 설정
        relevancy_threshold: float = None,
        fallback_faithfulness_min: float = None,
        fallback_faithfulness_max: float = None,
        fallback_relevancy_threshold: float = None
    ):
        """
        초기화
        
        Args:
            괄호 안 기본값은 settings에서 가져옴
            faithfulness_threshold: 충실도 통과 기준 (0.5)
            relevancy_threshold: 관련성 통과 기준 (0.7)
            fallback_faithfulness_min: fallback 충실도 최소 (0.1)
            fallback_faithfulness_max: fallback 충실도 최대 (0.5)
            fallback_relevancy_threshold: fallback 관련성 기준 (0.8)
        """
        # 파라미터 우선, 없으면 settings 사용 
        self.faithfulness_threshold = (
            faithfulness_threshold
            if faithfulness_threshold is not None
            else settings.evaluation.min_faithfulness
        )
        self.relevancy_threshold = (
            relevancy_threshold
            if relevancy_threshold is not None
            else settings.evaluation.min_relevancy
        )
        self.fallback_faithfulness_min = (
            fallback_faithfulness_min
            if fallback_faithfulness_min is not None
            else settings.evaluation.fallback_min_faithfulness
        )
        self.fallback_faithfulness_max = (
            fallback_faithfulness_max
            if fallback_faithfulness_max is not None
            else settings.evaluation.fallback_max_faithfulness
        )
        self.fallback_relevancy_threshold = (
            fallback_relevancy_threshold
            if fallback_relevancy_threshold is not None
            else settings.evaluation.fallback_min_relevancy
        )
        
        # LLM 설정 
        base_llm = ChatOpenAI(
            model_name=settings.openai.evaluation_model,
            temperature=0,
            openai_api_key=settings.openai.api_key,
            max_tokens=settings.openai.max_tokens_evaluation,
            request_timeout=60
        )

        self.llm = LangchainLLMWrapper(base_llm)
        self.metrics = [faithfulness, answer_relevancy]
        
        log.info(f"평가기 초기화 완료 (통과: faith>={faithfulness_threshold}, "
                f"rel>={relevancy_threshold})")
    
    def evaluate(
        self, 
        query: str, 
        answer: str, 
        documents: List[Document]
    ) -> EvaluationResult:
        """
        답변 평가 
        
        통과 조건: faithfulness >= 0.5 AND relevancy >= 0.7
        Fallback: 0.1 < faithfulness < 0.5 AND relevancy >= 0.8
        
        Args:
            query: 사용자 질문
            answer: 생성된 답변
            documents: 참조 문서
            
        Returns:
            EvaluationResult 객체
        """
        try:
            log.info("RAGAS 평가 시작")
            
            contexts = [doc.page_content for doc in documents]
            
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts]
            }
            dataset = Dataset.from_dict(data)
            
            evaluator_llm = self.llm
            
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=evaluator_llm
            )
            
            faith_score = float(result["faithfulness"])
            rel_score = float(result["answer_relevancy"])
            
            # 통과 조건: faithfulness >= 0.5 AND relevancy >= 0.7
            passed = (
                faith_score >= self.faithfulness_threshold and 
                rel_score >= self.relevancy_threshold
            )
            
            # Fallback 조건: 0.1 < faithfulness < 0.5 AND relevancy >= 0.8
            is_fallback = (
                self.fallback_faithfulness_min < faith_score < self.fallback_faithfulness_max and
                rel_score >= self.fallback_relevancy_threshold
            )
            
            feedback = f"충실도: {faith_score:.2f}, 관련성: {rel_score:.2f}"
            if passed:
                feedback += " ✅ 통과"
            elif is_fallback:
                feedback += " ⚠️ Fallback 대상"
            else:
                feedback += " ❌ 미달"
            
            log.info(f"평가 완료 - {feedback}")
            
            return EvaluationResult(
                faithfulness_score=faith_score,
                relevancy_score=rel_score,
                passed=passed,
                is_fallback=is_fallback,
                feedback=feedback
            )
        
        except Exception as e:
            log.exception(f"평가 오류: {str(e)}")
            return EvaluationResult(
                faithfulness_score=0.0,
                relevancy_score=0.0,
                passed=False,
                is_fallback=False,
                feedback=f"평가 오류: {str(e)}"
            )
