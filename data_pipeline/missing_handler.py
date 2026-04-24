# -*- coding: utf-8 -*-
"""
결측치 처리 모듈
GPT 기반 판시사항/판결요지 자동 생성 및 규칙 기반 병합
"""
import time
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from config.settings import settings
from utils.logger import log
from .preprocessor import is_nan_value




def validate_generated_content(content: str) -> Tuple[bool, str]:
    """
    GPT 생성 내용 검증
    
    Args:
        content: 생성된 내용
        
    Returns:
        (is_valid, error_message)
    """
    if not content or len(content) < 20:
        return False, "내용이 너무 짧습니다"
    
    if "형식 오류" in content or "오류 발생" in content:
        return False, "생성 오류 발생"
    
    if len(content.split()) < 5:
        return False, "내용 부족"
    
    return True, "OK"


class MissingDataHandler:
    """결측치 처리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=settings.openai.api_key,
            max_tokens=1000
        )
        log.info("결측치 처리 핸들러 초기화 완료")
    
    def identify_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 식별
        
        Args:
            df: 판례 데이터프레임
            
        Returns:
            결측치가 있는 행만 포함한 데이터프레임
        """
        missing_df = df[
            df['판례내용'].notnull() &
            df['판시사항'].isnull() &
            df['판결요지'].isnull()
        ].copy()
        
        log.info(f"결측치 발견: {len(missing_df)}개")
        return missing_df
    
    def generate_summary(self, case_content: str) -> Tuple[str, str]:
        """
        판례 내용에서 판시사항과 판결요지 생성
        
        Args:
            case_content: 판례 내용
            
        Returns:
            (판시사항, 판결요지)
        """
        try:
            system_message = SystemMessage(content="""
            당신은 법률 전문가입니다. 판례를 분석하여 판시사항과 판결요지를 
            정확하고 전문적으로 요약합니다.
            
            중요 규칙:
            1. 번호는 반드시 '[1]', '[2]' 형식 사용
            2. '1.', '2.' 형식은 절대 사용 금지
            3. 판시사항은 3개 이내의 핵심 쟁점만
            4. 판결요지는 200-300자 내외로 작성
            """)
            
            # 내용 길이 제한
            truncated_content = case_content[:8000]
            
            prompt = f"""
            다음 판례 내용을 분석하여 판시사항과 판결요지를 생성해주세요.
            
            [판례 내용]
            {truncated_content}
            
            [판시사항] 형식으로 판시사항을 작성하고,
            [판결요지] 형식으로 판결요지를 작성해주세요.
            
            판시사항: 법원이 판단한 핵심 법리/쟁점을 간결하게 (3개 이내)
            판결요지: 판단 근거와 결론을 200-300자로 요약
            """
            
            user_message = HumanMessage(content=prompt)
            response = self.llm([system_message, user_message])
            result = response.content
            
            # 판시사항과 판결요지 분리
            if "[판시사항]" in result and "[판결요지]" in result:
                parts = result.split("[판결요지]")
                holding = parts[0].replace("[판시사항]", "").strip()
                summary = parts[1].strip()
            else:
                holding = "형식 오류: " + result
                summary = "형식 오류: " + result
            
            # 검증
            is_valid_holding, _ = validate_generated_content(holding)
            is_valid_summary, _ = validate_generated_content(summary)
            
            if not (is_valid_holding and is_valid_summary):
                log.warning("생성된 내용 검증 실패")
            
            return holding, summary
        
        except Exception as e:
            log.error(f"판시사항/판결요지 생성 오류: {str(e)}")
            return f"오류 발생: {str(e)}", f"오류 발생: {str(e)}"
    
    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치를 GPT로 생성하여 채우기
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            결측치가 채워진 데이터프레임
        """
        # 결측치 식별
        missing_df = self.identify_missing(df)
        
        if len(missing_df) == 0:
            log.info("결측치 없음")
            return df
        
        # 결과 저장용 리스트
        holdings = []
        summaries = []
        indices = []
        
        # 각 행에 대해 판시사항/판결요지 생성
        success_count = 0
        error_count = 0
        
        for idx, row in tqdm(
            missing_df.iterrows(), 
            total=len(missing_df), 
            desc="판시사항/판결요지 생성 중"
        ):
            try:
                holding, summary = self.generate_summary(row['판례내용'])
                
                indices.append(idx)
                holdings.append(holding)
                summaries.append(summary)
            
                # 검증
                if "오류" not in holding and "오류" not in summary:
                    success_count += 1
                else:
                    error_count += 1
                
                # API 제한 방지
                time.sleep(1)
                
            except Exception as e:
                log.error(f"행 {idx} 처리 오류: {str(e)}")
                error_count += 1
        
        log.info(f"생성 완료: 성공 {success_count}개, 오류 {error_count}개")
        
        result_df = missing_df.copy()
        result_df.loc[indices, '판시사항'] = holdings
        result_df.loc[indices, '판결요지'] = summaries

        # 원본 데이터와 병합
        merged_df = self.merge_with_original(df, result_df)
        
        return merged_df
    
    def merge_with_original(
        self, 
        original_df: pd.DataFrame, 
        filled_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        원본 데이터와 생성 데이터 병합 (규칙 기반)
        
        규칙:
        1. 원본에 값이 있으면 원본 우선
        2. 원본이 NaN인 경우에만 생성값 사용
        
        Args:
            original_df: 원본 데이터프레임
            filled_df: 생성된 데이터프레임
            
        Returns:
            병합된 데이터프레임
        """
        key_col = '판례일련번호'
        
        # 병합할 컬럼 선택 (생성된 데이터만)
        filled_subset = filled_df[[key_col, '판시사항', '판결요지']].copy()
        filled_subset.columns = [key_col, '판시사항_generated', '판결요지_generated']

        # LEFT JOIN
        merged_df = original_df.merge(
            filled_subset, 
            on=key_col, 
            how='left',        )
        
        # 규칙 기반 결측치 채우기
        merged_df['판시사항'] = merged_df['판시사항'].fillna(
            merged_df['판시사항_generated']
        )
        merged_df['판결요지'] = merged_df['판결요지'].fillna(
            merged_df['판결요지_generated']
        )
        
        # 임시 컬럼 제거
        merged_df = merged_df.drop(
            columns=['판시사항_generated','판결요지_generated']
        )
        
        generated_count = (
            original_df['판시사항'].isnull() & 
            merged_df['판시사항'].notnull()
        ).sum()
        
        log.info(f"규칙 기반 병합 완료: {generated_count}개 생성값 사용")
        
        return merged_df
