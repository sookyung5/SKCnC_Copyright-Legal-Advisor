# -*- coding: utf-8 -*-
"""
의도 분석 모듈
질의 의도 분석 및 분류
"""
from typing import Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import settings
from utils import log
import json


@dataclass
class IntentResult:
    """의도 분석 결과"""
    is_copyright_related: bool  # 저작권법 관련 여부
    doc_type: str  # 문서 유형 (법령/판례/둘다)
    specific_article: Optional[str]  # 특정 조문
    is_core_needed: bool  # 핵심 필요성
    confidence: float  # 신뢰도


class IntentAnalyzer:
    """질의 의도 분석기"""
    
    def __init__(self):
        """초기화"""
        self.llm = ChatOpenAI(
            model_name=settings.openai.evaluation_model, 
            temperature=0,
            openai_api_key=settings.openai.api_key,
            max_tokens=settings.openai.max_tokens_intent, # 500 tokens
            request_timeout=30
        )
        
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""다음 질문을 분석하여 JSON 형식으로 답변하세요:

질문: {query}

분석 기준:
1. is_copyright_related: 저작권법 관련 질문인가? (true/false)
2. doc_type: 필요한 문서 유형 ("법령", "판례", "둘다")
3. specific_article: 특정 조문 언급 여부 (예: "제30조", 없으면 null)
4. is_core_needed: 핵심 판례/법령이 필요한가? (true/false)
5. confidence: 분석 신뢰도 (0.0~1.0)

반드시 다음 JSON 형식으로만 답변하세요:
{{
    "is_copyright_related": true,
    "doc_type": "판례",
    "specific_article": "제30조",
    "is_core_needed": true,
    "confidence": 0.95
}}"""
        )
        
        log.info("의도 분석기 초기화 완료 (max_tokens={})".format(
            settings.openai.max_tokens_intent
        ))
    
    def analyze(self, query: str) -> IntentResult:
        """
        질의 의도 분석
        
        Args:
            query: 사용자 질문
            
        Returns:
            IntentResult 객체
        """
        try:
            prompt = self.prompt.format(query=query)
            response = self.llm.predict(prompt)
            
            # JSON 파싱 
            result = json.loads(response)
            
            return IntentResult(
                is_copyright_related=result.get("is_copyright_related", True),
                doc_type=result.get("doc_type", "둘다"),
                specific_article=result.get("specific_article"),
                is_core_needed=result.get("is_core_needed", False),
                confidence=result.get("confidence", 0.8)
            )
        
        except Exception as e:
            log.error(f"의도 분석 오류: {str(e)}")
            # 기본값 반환
            return IntentResult(
                is_copyright_related=True,
                doc_type="둘다",
                specific_article=None,
                is_core_needed=False,
                confidence=0.5
            )
