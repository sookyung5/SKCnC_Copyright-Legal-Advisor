# -*- coding: utf-8 -*-
"""
설정 관리 모듈
환경 변수를 통한 안전한 설정 관리 (재시도 로직 + Voyage Reranker 포함)
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class OpenAIConfig(BaseModel):
    """OpenAI API 설정"""
    api_key: str = Field(..., description="OpenAI API 키")
    embedding_model: str = Field(default="text-embedding-3-large")
    llm_model: str = Field(default="gpt-4-turbo")
    evaluation_model: str = Field(default="gpt-4o", description="평가용 모델")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    max_tokens_generation: int = Field(default=1000, description="답변 생성 최대 토큰")
    max_tokens_intent: int = Field(default=500, description="의도 분석 최대 토큰")
    max_tokens_evaluation: int = Field(default=1000, description="평가 최대 토큰")
    max_tokens_multiquery: int = Field(default=300, description="멀티쿼리 생성 최대 토큰")

class VoyageConfig(BaseModel):
    """Voyage AI 설정"""
    api_key: str = Field(..., description="Voyage AI API 키")
    reranker_model: str = Field(default="rerank-2.5")
    top_k: int = Field(default=5, gt=0, description="리랭킹 후 상위 k개")


class ChunkingConfig(BaseModel):
    """청킹 설정"""
    # 판례 청킹
    case_separators: List[str] = Field(default=[
        "\n\n", ".\n", ". ",
        "【이유】", "【주문】",
        "\n1. ", "\n2. ", "\n3. ",
        "\n가. ", "\n나. ", "\n다. ", "\n라. ",
        ") ", ", ", " "
    ])
    case_chunk_size: int = Field(default=500, gt=0)
    case_chunk_overlap: int = Field(default=100, ge=0)
    case_min_chunk_size: int = Field(default=300, gt=0)
    
    # 법령 청킹
    law_separators: List[str] = Field(
        default=[
            "\n   ",    # 호 단위
            "\n     ",  # 목 단위
            "\n",       # 줄바꿈
            ". ",       # 문장
            ", ",       # 구절
            " ",        # 단어
            ""          # 문자
        ],
        desc="법령 청킹 구분자 (조문 통합 후 사용)"
    )
    law_chunk_size: int = Field(default=500, gt=0)
    law_chunk_overlap: int = Field(default=125, ge=0)
    law_min_chunk_size: int = Field(default=250, gt=0)
    
    # 부칙 청킹
    addendum_separators: List[str] = Field(
        default=[
            "\n  ",     # 항 시작
            "\n    ",   # 호 시작
            "\n",       # 줄바꿈
            ". ",       # 문장
            ", ",       # 구절
            " ",        # 단어
            ""          # 문자
        ],
        desc="부칙 청킹 구분자 (조문 통합 후 사용)"
    )
    addendum_chunk_size: int = Field(default=500, gt=0)
    addendum_chunk_overlap: int = Field(default=100, ge=0)
    addendum_min_chunk_size: int = Field(default=100, gt=0)

    # 시행령 청킹
    enforcement_separators: List[str] = Field(
        default=[
            "\n   ",    # 호 단위
            "\n     ",  # 목 단위
            "\n",       # 줄바꿈
            ". ",       # 문장
            ", ",       # 구절
            " ",        # 단어
            ""          # 문자
        ],
        desc="시행령 청킹 구분자 (조문 통합 후 사용)"
    )
    enforcement_chunk_size: int = Field(default=500, gt=0)
    enforcement_chunk_overlap: int = Field(default=100, ge=0)
    enforcement_min_chunk_size: int = Field(default=100, gt=0)

    # 공통 설정
    min_chunk_size: int = Field(default=100, gt=0)


class RetrievalConfig(BaseModel):
    """검색 설정"""
    # 기본 검색 파라미터
    top_k: int = Field(default=10, gt=0, description="초기 검색 문서 수")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # 재시도별 검색 파라미터
    retry_0_k: int = Field(default=10, description="1차 시도 검색 수")
    retry_1_k: int = Field(default=15, description="2차 시도 검색 수")
    retry_2_k: int = Field(default=20, description="3차 시도 검색 수")
    
    # 리랭킹
    use_reranker: bool = Field(default=True, description="리랭커 사용 여부")
    rerank_top_k: int = Field(default=5, gt=0, description="재순위화 후 문서 수")


class EvaluationConfig(BaseModel):
    """평가 설정"""
    min_passing_score: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="통과 기준 점수"
    )
    faithfulness_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    relevancy_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # 통과 조건
    min_faithfulness: float = Field(default=0.5, ge=0.0, le=1.0)
    min_relevancy: float = Field(default=0.7, ge=0.0, le=1.0)

    # Fallback 조건
    fallback_min_faithfulness: float = Field(default=0.1, ge=0.0, le=1.0)
    fallback_max_faithfulness: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_min_relevancy: float = Field(default=0.8, ge=0.0, le=1.0)

class RetryConfig(BaseModel):
    """재시도 설정"""
    max_retry_count: int = Field(
        default=3, 
        gt=0, 
        le=5,
        description="최대 재시도 횟수"
    )
    retry_on_evaluation_fail: bool = Field(
        default=True,
        description="평가 실패시 재시도 여부"
    )
    retry_on_empty_result: bool = Field(
        default=True,
        description="검색 결과 없을 때 재시도 여부"
    )


class PathConfig(BaseModel):
    """경로 설정"""
    data_dir: Path = Field(default=BASE_DIR / "data")
    vectorstore_path: Path = Field(default=BASE_DIR / "data" / "vectorstore")
    cache_dir: Path = Field(default=BASE_DIR / "cache")
    log_dir: Path = Field(default=BASE_DIR / "logs")
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        for path in [self.data_dir, self.vectorstore_path, 
                     self.cache_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Settings:
    """통합 설정"""
    
    def __init__(self):
        # OpenAI 설정
        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
            evaluation_model=os.getenv("EVALUATION_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),

            # max_tokens 환경변수
            max_tokens_generation=int(os.getenv("MAX_TOKENS_GENERATION", "1000")),
            max_tokens_intent=int(os.getenv("MAX_TOKENS_INTENT", "500")),
            max_tokens_evaluation=int(os.getenv("MAX_TOKENS_EVALUATION", "1000")),
            max_tokens_multiquery=int(os.getenv("MAX_TOKENS_MULTIQUERY", "300"))
        )
        
        # Voyage 리랭커 설정
        self.voyage = VoyageConfig(
            api_key=os.getenv("VOYAGE_API_KEY", ""),
            reranker_model=os.getenv("VOYAGE_RERANKER_MODEL", 
                                    "rerank-2.5"),
            top_k=int(os.getenv("VOYAGE_TOP_K", "5"))
        )
    

        # 청킹 설정
        self.chunking = ChunkingConfig(
            case_chunk_size=int(os.getenv("CASE_CHUNK_SIZE", "600")),
            case_chunk_overlap=int(os.getenv("CASE_CHUNK_OVERLAP", "150")),
            law_chunk_size=int(os.getenv("LAW_CHUNK_SIZE", "500")),
            law_chunk_overlap=int(os.getenv("LAW_CHUNK_OVERLAP", "125"))
        )
        
        # 검색 설정
        self.retrieval = RetrievalConfig(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
            use_reranker=os.getenv("USE_RERANKER", "true").lower() == "true",
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
            retry_0_k=int(os.getenv("RETRY_0_K", "10")),
            retry_1_k=int(os.getenv("RETRY_1_K", "15")),
            retry_2_k=int(os.getenv("RETRY_2_K", "20"))
        )
        
        # 평가 설정
        self.evaluation = EvaluationConfig(
            min_passing_score=float(os.getenv("MIN_PASSING_SCORE", "0.7")),
            faithfulness_weight=float(os.getenv("FAITHFULNESS_WEIGHT", "0.5")),
            relevancy_weight=float(os.getenv("RELEVANCY_WEIGHT", "0.5"))
        )
        
        # 재시도 설정
        self.retry = RetryConfig(
            max_retry_count=int(os.getenv("MAX_RETRY_COUNT", "3")),
            retry_on_evaluation_fail=os.getenv("RETRY_ON_EVAL_FAIL", "true").lower() == "true",
            retry_on_empty_result=os.getenv("RETRY_ON_EMPTY", "true").lower() == "true"
        )
        
        # 경로 설정
        self.paths = PathConfig(
            data_dir=Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))),
            vectorstore_path=Path(os.getenv("VECTORSTORE_PATH", 
                                          str(BASE_DIR / "data" / "vectorstore"))),
            cache_dir=Path(os.getenv("CACHE_DIR", str(BASE_DIR / "cache"))),
            log_dir=Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))
        )
        
        # 디렉토리 생성
        self.paths.create_directories()
        
        # 로그 레벨
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API 키 검증
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """API 키 검증"""
        if not self.openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하세요."
            )
        
        if self.retrieval.use_reranker and not self.voyage.api_key:
            raise ValueError(
                "리랭커 사용이 활성화되었으나 VOYAGE_API_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하거나 USE_RERANKER=false로 설정하세요."
            )
    
    @property
    def MAX_RETRY_COUNT(self) -> int:
        """하위 호환성을 위한 속성 (이전 코드와 호환)"""
        return self.retry.max_retry_count


# 전역 설정 인스턴스
settings = Settings()
