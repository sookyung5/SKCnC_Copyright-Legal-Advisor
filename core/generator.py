# -*- coding: utf-8 -*-
"""
답변 생성 모듈
LLM을 사용한 답변 생성
"""
import re
from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import settings
from utils import log


class AnswerGenerator:
    """답변 생성기"""
    
    def __init__(self):
        """초기화"""
        self.llm = ChatOpenAI(
            model_name=settings.openai.llm_model, # gpt-4-turbo
            temperature=settings.openai.temperature, # 0.3
            openai_api_key=settings.openai.api_key,
            max_tokens=settings.openai.max_tokens_generation, # 1000 tokens
            request_timeout=60
        )
        
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""당신은 저작권법 전문 AI 법무자문 챗봇입니다. 

질문: {query}

참조 문서:
{context}

답변 작성 규칙:
1. 질문에 직접적으로 답변
2. 관련 법령 조문 인용 (예: 저작권법 제X조)
3. 판례가 있다면 사건번호 명시
4. 명확하고 구체적으로 설명
5. 출처를 반드시 표시


답변:"""
        )
        
        log.info("답변 생성기 초기화 완료 (model={}, max_tokens={})".format(
            settings.openai.llm_model,
            settings.openai.max_tokens_generation
        ))
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """
        답변 생성
        
        Args:
            query: 사용자 질문
            documents: 검색된 문서 리스트
            
        Returns:
            생성된 답변
        
        Raises:
            ValueError: 문서가 비어있을 때
            Exception: LLM 호출 오류 
        """
        if not documents:
            log.warning("문서가 없어 답변 생성 불가")
            raise ValueError("검색된 문서가 없습니다.")

        try:
            # 컨텍스트 포맷팅
            context = self._format_documents(documents)
            
            if not context or not context.strip():
                log.warning("문서 내용이 비어있음")
                raise ValueError("문서 내용이 없습니다.")

            # 프롬프트 생성
            prompt = self.prompt.format(query=query, context=context)

            # LLM 호출
            answer = self.llm.predict(prompt)

            log.info(f"답변 생성 완료 (길이: {len(answer)}자)")
            return answer
        
        except ValueError:
            raise
        except Exception as e:
            log.exception(
                f"답변 생성 오류: {str(e)} "
                f"(query: {query[:50]}, doc_count: {len(documents)})")
            raise


    def _format_documents(self, documents: List[Document]) -> str:
        """
        문서를 프롬프트용 텍스트로 변환

        Args:
            documents: 검색된 문서 리스트
        
        Returns:
            포맷팅된 컨텍스트
        """
        formatted = []

        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            content = doc.page_content

            doc_type = (
                metadata.get('doc_type') or
                metadata.get('문서유형', 'unknown')
            )
            
            # 문서 타입별 헤더 
            if doc_type in ['case', '판례']:
                # 판례
                case_num = metadata.get('사건번호', 'N/A')
                decision_date = metadata.get('선고일자', 'N/A')
                court = metadata.get('법원명', '')

                header = f"[판례 {i}] {court} {decision_date} 선고 {case_num} 판결"
            elif doc_type in ['law', '법령']:
                # 조문
                article_num = metadata.get('조문번호', '')
                article_title = metadata.get('조문제목', '')
                included_hangs = metadata.get('포함항', [])
                included_hos = metadata.get('포함호', [])

                header = f"[법령 {i}] 저작권법 제{article_num}조"
                if article_title:
                    header += f" ({article_title})"

                if included_hangs:
                    header += f" 제{','.join(map(str, included_hangs))}항"
                
                if included_hos:
                    header += f" {','.join(map(str, included_hos))}호"
            
            elif doc_type in ['addendum', '부칙']:
                # 부칙
                pub_num = metadata.get('공포번호', '')
                law_title = metadata.get('법률제목', '저작권법')
                article_num = metadata.get('부칙_조문번호', '')
                article_title = metadata.get('부칙_조문제목', '')
                included_hangs = metadata.get('포함항', [])
                included_hos = metadata.get('포함호', [])
                
                header = f"[부칙 {i}] 법률 제{pub_num}호 {law_title} 부칙 제{article_num}조"
                if article_title:
                    header += f" ({article_title})"
                
                if included_hangs:
                    header += f" 제{','.join(map(str, included_hangs))}항"
                
                if included_hos:
                    header += f" {','.join(map(str, included_hos))}호"
                
            elif doc_type in ['enforcement_decree', '시행령']:
                # 시행령
                article_num = metadata.get('시행령_조문번호', '')
                article_title = metadata.get('시행령_조문제목', '')
                included_hangs = metadata.get('포함항', [])
                included_hos = metadata.get('포함호', [])
                
                header = f"[시행령 {i}] 저작권법 시행령 제{article_num}조"
                if article_title:
                    header += f" ({article_title})"
                
                if included_hangs:
                    header += f" 제{','.join(map(str, included_hangs))}항"
                
                if included_hos:
                    header += f" {','.join(map(str, included_hos))}호"
                    
            else:
                # 기타
                header = f"[문서 {i}]"

            formatted.append(f"{header}\n{content}\n")

        return "\n".join(formatted)


class RelatedQuestionGenerator:
    """연관 질문 생성기"""
    
    def __init__(self):
        """초기화"""
        self.llm = ChatOpenAI(
            model_name=settings.openai.evaluation_model,
            temperature=0.7,
            openai_api_key=settings.openai.api_key,
            max_tokens=300,  # 연관 질문은 짧게
            request_timeout=30
        )
        
        self.prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template="""다음 질문과 답변을 바탕으로 연관된 질문 3개를 생성하세요.

            질문: {query}
            답변: {answer}

            반드시 아래 형식으로만 출력하세요. 다른 텍스트는 포함하지 마세요.
            1. [첫 번째 질문]
            2. [두 번째 질문]
            3. [세 번째 질문]"""
        )

        log.info("연관 질문 생성기 초기화 완료")

    def generate(self, query: str, answer: str) -> List[str]:
        """
        연관 질문 생성
        
        Args:
            query: 원본 질문
            answer: 생성된 답변
            
        Returns:
            연관 질문 리스트 (3개)
        """
        try:
            prompt = self.prompt.format(query=query, answer=answer)
            response = self.llm.predict(prompt)
            
            # 파싱
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                match = re.match(r'^[1-3][.)]\s*(.+)', line)
                if match:
                    questions.append(match.group(1).strip())
            
            log.info(f"연관 질문 생성 완료: {len(questions)}개")
            return questions[:3]  # 최대 3개

            
        except Exception as e:
            log.exception(
                f"연관 질문 생성 오류: {str(e)}")
            return []
