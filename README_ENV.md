# 환경변수 설정 가이드


---

## 📁 목차

- [빠른 시작](#-빠른-시작)
- [필수 환경변수](#-필수-환경변수)
- [OpenAI 설정](#-openai-설정)
- [Voyage AI 설정](#-voyage-ai-설정)
- [청킹 설정](#-청킹-설정)
- [검색 설정](#-검색-설정)
- [평가 설정](#-평가-설정)
- [재시도 설정](#-재시도-설정)
- [추천 프리셋](#-추천-프리셋)
- [문제 해결](#-문제-해결)

---

## 🚀 빠른 시작

### 1. `.env` 파일 생성

```bash
# 프로젝트 루트 디렉토리에서
cp .env.example .env
```

### 2. 필수 환경 변수 입력 

```env
OPENAI_API_KEY=your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key
```

### 3. 실행

```bash
streamlit run app.py 
# 또는 개발용: python main.py
```

나머지 설정은 기본값 사용!

---

## 🔑 필수 환경변수 상세 

### OPENAI_API_KEY

- 용도: OpenAI API 접근 키 (임베딩 모델 및 LLM호출시 필요)
- 발급: https://platform.openai.com > API Keys > Create new secret key
- 보안 권장:
    - `.env` 커밋 금지
    - 3개월 주기 재발급
    - Billing 사용량 모니터링 

---

### Voyage_API_KEY

- 용도: Voyage AI Reranker API 접근 키
- 발급 방법: https://www.voyageai.com/ > 로그인 > Create new secret key 
- 리랭커 미사용 시 불필요:  
    ```env
    USE_RERANKER = false
    ``` 

---

## 🤖 OPENAI 설정
모델 성능·비용·정확도 균형을 위한 핵심 조정 파라미터입니다.

```env
EMBEDDING_MODEL = text-embedding-3-large    # 검색용 임베딩 모델 
LLM_MODEL = gpt-4-turbo                     # 답변 생성 모델 
EVALUATION_MODEL = gpt-4o                   # RAGAS 평가·의도분석 모델 

LLM_TEMPERATURE = 0.3           # 답변 창의성 
MAX_TOKENS_GENERATION =1000     # 답변 생성 최대 토큰 
MAX_TOKENS_INTENT=500           # 의도 분석 최대 토큰 
MAX_TOKENS_MULTIQUERY=300       # 멀티쿼리 생성 최대 토큰 
```


---

## 🔍 Voyage 리랭커 설정

리랭커는 검색된 문서를 "질문 관련도 기준"으로 재정렬해 답변 품질을 크게 향상시킵니다. 

```env
VOYAGE_RERANKER_MODEL=rerank-2.5             # 한국어 지원 리랭커
USE_RERANKER=TRUE                            # 리랭커 활성화 여부
VOYAGE_TOP_K=5                                 # 리랭킹 후 선택 문서 수
```

---

## ✂️ 청킹 설정

임베딩 모델의 토큰 한도 내에서 문서를 안정적으로 나누기 위한 설정입니다. 
(단위: 문자 수/토큰 기준으로 전환 시 `chunk ≈ tokenx2~3)`) 

```env
CASE_CHUNK_SIZE=600     # 판례 청크 크기 (기본값)
CASE_CHUNK_OVERLAP=150  # 판례 청크 오버랩 (기본값)
LAW_CHUNK_SIZE=500      # 법령 청크 크기 (기본값)
LAW_CHUNK_OVERLAP=125   # 법령 청크 오버랩 (기본값)
```

**가이드라인**:
- 작게 (300-400): 정밀한 검색, 많은 청크
- 중간 (600): 균형잡힌 설정 (권장)
- 크게 (800-1000): 넓은 컨텍스트, 적은 청크


---

## 🔍 검색 설정

검색(k) 값과 리랭킹 문서 수를 제어합니다. k가 크면 리콜이 증가하지만 속도가 저하될 수 있습니다. 

```env
RETRIEVAL_TOP_K=10  # 초기 검색 문서 개수 
RETRY_0_K=10        # 재시도 0회차 검색 개수 
RETRY_1_K=15        # 재시도 1회차 검색 개수 
RETRY_2_K=20        # 재시도 2회차 검색 개수 
RERANK_TOP_K=5      # 리랭킹 후 선택할 문서 개수 
```

### 검색 전략

```env
Retry 0 → k=10  (빠른 1차 응답)
Retry 1 → k=15  (중간 탐색)
Retry 2 → k=20  (광범위 탐색)
```

---

## ✅ 평가 설정

RAGAS 기반으로 충실성과 답변 관련성을 평가해 자동 품질 보증합니다. 

```env
MIN_PASSING_SCORE=0.7

FAITHFULNESS_WEIGHT=0.5 # 충실성 가중치
RELEVANCY_WEIGHT=0.5    # 관련성 가중치 
```

**조정 가이드**:
- `0.6`: 관대한 기준 (통과율 ↑)
- `0.7`: 균형잡힌 기준 (권장)
- `0.9`: 엄격한 기준 (통과율 ↓)


---

## 🔄 재시도 설정

검색 실패 및 평가 불합격 시 자동으로 재시도하는 로직을 제어합니다. 

```env
MAX_RETRY_COUNT=3       # 최대 재시도 횟수 
RETRY_ON_EVAL_FAIL=true # 평가 실패 시 재시도 활성화 
RETRY_ON_EMPTY=true     # 빈 결과 시 재시도 활성화 
```

**통계**:
재시도 및 누적 통과율 측정 예정


---

## 🎯 추천 프리셋

### 로컬 개발 환경

```env
# API 키
OPENAI_API_KEY=your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key

# 빠른 응답 (비용 절감)
LLM_MODEL=gpt-3.5-turbo
EVALUATION_MODEL=gpt-4o
USE_RERANKER=false
MAX_RETRY_COUNT=1

# 로깅
LOG_LEVEL=DEBUG
```


---

### 프로덕션 환경

```env
# API 키
OPENAI_API_KEY=your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key

# 고품질 답변
LLM_MODEL=gpt-4-turbo
EVALUATION_MODEL=gpt-4o
USE_RERANKER=true
MAX_RETRY_COUNT=3

# 최적화
RETRIEVAL_TOP_K=10
RETRY_0_K=10
RETRY_1_K=15
RETRY_2_K=20

# 로깅
LOG_LEVEL=INFO
```

---

### 데모/발표 환경

```env
# API 키
OPENAI_API_KEY=your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key

# 안정적 답변
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.1
EVALUATION_MODEL=gpt-4o
USE_RERANKER=true
MAX_RETRY_COUNT=2

# 로깅
LOG_LEVEL=WARNING
```

---

### 비용 최적화 환경

```env
# API 키
OPENAI_API_KEY=your-openai-api-key

# 최소 비용
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
EVALUATION_MODEL=gpt-4o
USE_RERANKER=false
MAX_RETRY_COUNT=1

# 검색 최소화
RETRIEVAL_TOP_K=5
RERANK_TOP_K=3
```


---

## 🛠️ 문제 해결 (FAQ)

### Q1:` "OPENAI_API_KEY not found"`

```bash
# .env 파일 확인
cat .env | grep OPENAI_API_KEY

# 키가 없으면 추가
echo "OPENAI_API_KEY=your-openai-api-key" >> .env
```

---

### Q2: `"VOYAGE_API_KEY required"` 


```env
# 키 추가
VOYAGE_API_KEY=your-voyage-api-key

# 또는 리랭커 비활성화
USE_RERANKER=false
```

---

### Q3: 답변이 너무 느림 (10초 이상)


```env
# 재시도 감소
MAX_RETRY_COUNT=1

# k값 감소
RETRIEVAL_TOP_K=5
RETRY_0_K=5
RETRY_1_K=10

# 리랭커 비활성화
USE_RERANKER=false
```

---

### Q4: 비용이 너무 높음

```env
# 저렴한 모델로 변경
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# 리랭커 비활성화
USE_RERANKER=false

# 재시도 감소
MAX_RETRY_COUNT=1
```
---

### Q5: 평가가 너무 엄격함 (재시도 과다)

**해결**:
```env
# 평가 기준 완화
MIN_PASSING_SCORE=0.6

# 재시도 감소
MAX_RETRY_COUNT=2
```

---

### Q6: 메모리 부족

**해결**:
```env
# 청크 크기 감소
CASE_CHUNK_SIZE=400
LAW_CHUNK_SIZE=300

# 검색 개수 감소
RETRIEVAL_TOP_K=5
RETRY_0_K=5
RETRY_1_K=8
RETRY_2_K=10
```

---


## 🔒 보안 가이드

### .gitignore 설정

```gitignore
# 환경변수 (필수!)
.env
.env.local
.env.*.local

# API 키
*.key
*.secret

# 백업
.env.backup
```

### 주의사항 

1. **`.env` 파일 절대 커밋하기 금지!**
```bash
# .env 파일을 실수로 추가했다면
git rm --cached .env
git commit -m "Remove .env"
```

2. **api키는 환경변수로만 사용**
```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

3. **.env.example만 공유**
```bash
# .env를 .env.example로 복사
cp .env .env.example

# .env.example에서 실제 키 제거
# OPENAI_API_KEY=your-key-here → OPENAI_API_KEY=
```

4. **권한 설정** (Linux/Mac)
```bash
chmod 600 .env  # 소유자만 읽기/쓰기
```

---

## 📈 모니터링

### OpenAI 비용
OpenAI Dashboard에서 사용량 확인:
- https://platform.openai.com/usage

### 성능 모니터링

로그에서 성능 확인:
```bash
grep "Total time" logs/app.log | tail -20
```

---

## 📚 추가 참고

- **문서**: [README.md](./README.md), [MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)
- **이슈**: GitHub Issues

---

<div align="center">

**[⬆ 맨 위로](#환경변수-설정-가이드)**

</div>
