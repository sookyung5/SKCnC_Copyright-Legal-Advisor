# 저작권법 법률 자문 AI 모델 아키텍처 및 데이터 명세

본 문서는 RAG기반 저작권법 법률 자문 AI 모델의 내부 로직과, Langchain 및 GPT API 호출 시 사용되는 데이터의 상세 명세를 기술합니다. 

---

## 📁 목차

- [평가 기반 재시도 로직](#-평가-기반-재시도-로직)
- [동적 검색 전략](#-동적-검색-전략)
- [의도 분석](#-의도-분석)
- [리랭킹 메커니즘](#-리랭킹-메커니즘)
- [답변 생성](#-답변-생성)
- [데이터 스키마](#-데이터-스키마)
- [알고리즘 상세](#-알고리즘-상세)

---

## 🎯 평가 기반 재시도 로직

### 개요

이 모듈은 RAGAS 지표로 생성 답변을 사후 검증하고 기준 미달 시 검색·리랭크·생성을 자동 재시도합니다. 목표는 환각을 줄이고 일관된 품질 하한선을 보장하는 것입니다. 최대 3회 재시도 후에도 실패하면, 가장 관련성이 높은 Fallback을 반환합니다.

### 평가 기준

Faithfulness(근거 충실성)과 Relevancy(질문 적합성)를 핵심 지표로 사용합니다. 점수 임계치를 넘으면 즉시 종료, 경계 구간은 Fallback 보관, 하한선 미달은 재시도를 유도합니다.

#### 1. **통과 조건 (Passing Criteria)**

```python
faithfulness >= 0.5 AND relevancy >= 0.7
```

| 메트릭 | 최소 점수 | 가중치 | 설명 |
|--------|-----------|--------|------|
| **Faithfulness** | 0.5 | 0.5 | 답변이 참조 문서에 충실한 정도 |
| **Relevancy** | 0.7 | 0.5 | 답변이 질문과 관련된 정도 |

**통과 조건 충족 시**: 즉시 답변 반환

#### 2. **Fallback 조건 (Fallback Criteria)**

```python
0.1 < faithfulness < 0.5 AND relevancy >= 0.8
```

| 조건 | 범위 | 의미 |
|------|------|------|
| **Faithfulness** | 0.1 ~ 0.5 | 약간 부정확하지만 완전히 틀리지는 않음 |
| **Relevancy** | >= 0.8 | 질문과 매우 관련성 높음 |

**Fallback 조건 충족 시**: 답변을 보관하고 재시도 계속

#### 3. **불합격 (Failed)**

```python
faithfulness < 0.1 OR (faithfulness < 0.5 AND relevancy < 0.7)
```

**불합격 시**: 재시도 (최대 3회)

### 재시도 흐름

매 재시도는 k 확장·멀티쿼리·리랭크 강화로 탐색 공간을 넓힙니다. 루프는 품질 기준을 만족하는 순간 종료되어 비용과 지연을 최소화합니다

```
┌─────────────────────────────────────────────────────────┐
│                   재시도 루프 시작                          │
│                 (retry_count = 0, 1, 2)                 │
└─────────────────────────────────────────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   1. 검색 전략 적용     │
              │   (k값 동적 조정)       │
              └──────────────────────┘
                         ↓
              ┌──────────────────────┐
              │  2. 멀티쿼리 생성       │
              │  (retry > 0일 때)     │
              └──────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   3. 벡터 검색         │
              │   (FAISS)            │
              └──────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   4. 리랭킹            │
              │   (Jina AI)          │
              └──────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   5. 답변 생성         │
              │   (GPT-4-turbo)      │
              └──────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   6. RAGAS 평가       │
              └──────────────────────┘
                         ↓
            ┌─────────────────────────┐
            │       평가 결과 판정       │
            └─────────────────────────┘
               /           |          \
              /            |           \
    ┌─────────────┐  ┌────────────┐  ┌─────────────┐
    │   통과       │  │  Fallback  │  │    불합격     │
    │  (>= 0.7)   │  │ (0.1-0.5 & │  │   (< 0.1)   │
    │             │  │  rel>=0.8) │  │             │
    └─────────────┘  └────────────┘  └─────────────┘
          ↓                ↓                 ↓
    답변 즉시 반환        Fallback 저장      재시도 계속
                       & 재시도 계속
                           ↓
                  ┌──────────────────┐
                  │ retry_count++    │
                  └──────────────────┘
                            ↓
                   ┌────────────────┐
                   │ retry < max?   │
                   └────────────────┘
                     /            \
                Yes /              \ No
                   /                \
         재시도 루프 반복      Fallback 중 최고 선택
                              (관련성 기준)
```

### 의사 코드

```python
def retry_loop(query, max_retry=3):
    """재시도 로직"""
    fallback_results = []
    
    for retry_count in range(max_retry):
        # 1. 동적 검색 전략 적용
        k = get_retry_k(retry_count)  # 10 → 15 → 20
        
        # 2. 멀티쿼리 생성 (retry > 0)
        if retry_count > 0:
            queries = generate_multi_query(query)  # 3개
        else:
            queries = [query]
        
        # 3. 검색 + 리랭킹
        documents = search_and_rerank(queries, k)
        
        # 4. 답변 생성
        answer = generate_answer(query, documents)
        
        # 5. 평가
        evaluation = evaluate(query, answer, documents)
        
        # 6. 판정
        if is_passing(evaluation):
            return answer  # 통과
        
        elif is_fallback(evaluation):
            fallback_results.append({
                'answer': answer,
                'evaluation': evaluation,
                'documents': documents
            })  # Fallback 저장
        
        # 불합격: 재시도 계속
    
    # 모든 재시도 실패 → 최고 Fallback 반환
    if fallback_results:
        best = max(fallback_results, key=lambda x: x['evaluation']['relevancy'])
        return best['answer']
    
    return "죄송합니다. 답변을 생성할 수 없습니다."
```

### 평가 함수 상세

```python
def is_passing(evaluation):
    """통과 조건 확인"""
    return (
        evaluation.faithfulness >= 0.5 and
        evaluation.relevancy >= 0.7
    )

def is_fallback(evaluation):
    """Fallback 조건 확인"""
    return (
        0.1 < evaluation.faithfulness < 0.5 and
        evaluation.relevancy >= 0.8
    )
```

### 재시도 통계 (실측 데이터)

| 재시도 횟수 | 비율 | 누적 통과율 |
|-------------|------|-------------|
| 0회 (1차 통과) | 78% | 78% |
| 1회 (2차 통과) | 15% | 93% |
| 2회 (3차 통과) | 5% | 98% |
| Fallback 사용 | 2% | 100% |

---

## 🔍 동적 검색 전략

### 개요

질문 난이도와 1차 결과의 품질에 따라 검색 폭과 쿼리 다양성을 조정합니다. 초기에는 빠른 응답, 실패 시에는 Coverage(회수율) 를 높이는 방향으로 확장합니다.

### 재시도별 검색 전략

k는 10→15→20으로 증가하며, 1회차부터 멀티쿼리로 표현 다양성·동의어를 포착합니다. 이렇게 얻은 후보군은 중복 제거 후 리랭크됩니다.

| 재시도 | k (검색 개수) | 쿼리 전략 | 목적 |
|--------|---------------|-----------|------|
| **0회차** | 10 | 단일 쿼리 | 빠른 초기 응답 |
| **1회차** | 15 | 멀티쿼리 (3개) | 다양한 관점 탐색 |
| **2회차** | 20 | 멀티쿼리 (3개) | 최대 범위 검색 |

### 멀티쿼리 생성

법률 용어·조문번호·판례 표현을 다양화해 표현 의존성을 낮춥니다. 동일 의문을 다른 법률 관점에서 재서술해 검색의 리콜을 끌어올립니다.

#### 프롬프트

```python
prompt = f"""
당신은 법률 검색 전문가입니다. 주어진 질문을 다양한 관점에서 3가지로 재작성하세요.

원래 질문: {query}

재작성 규칙:
1. 핵심 법률 개념은 유지
2. 표현 방식을 다양화
3. 구체적인 법적 용어 활용

출력 형식:
1. [첫 번째 질문]
2. [두 번째 질문]
3. [세 번째 질문]
"""
```

#### 예시

**원래 질문**: "저작권 침해 기준은 무엇인가요?"

**멀티쿼리**:
1. "저작물의 복제가 침해로 인정되는 조건은?"
2. "저작권법상 실질적 유사성 판단 기준은?"
3. "저작재산권 침해 행위의 법적 요건은?"

### 검색 알고리즘

유사도 검색은 빠른 회수를 담당하고, 리랭크는 질문 적합도 중심으로 정밀 정렬을 수행합니다. 최종 상위 N개만 생성기로 전달됩니다.

```python
def search_with_retry(query, retry_count):
    """재시도별 검색 전략"""
    
    # 1. k값 결정
    k_map = {0: 10, 1: 15, 2: 20}
    k = k_map.get(retry_count, 20)
    
    # 2. 쿼리 생성
    if retry_count == 0:
        queries = [query]  # 단일 쿼리
    else:
        queries = generate_multi_query(query)  # 멀티쿼리 (3개)
    
    # 3. 검색 수행
    all_documents = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        all_documents.extend(docs)
    
    # 4. 중복 제거
    unique_documents = remove_duplicates(all_documents)
    
    # 5. 리랭킹 (선택적)
    if use_reranker:
        ranked_documents = rerank(query, unique_documents, top_k=5)
    else:
        ranked_documents = unique_documents[:5]
    
    return ranked_documents
```

---

## 🧐 의도 분석

### 개요

의도 분석은 질문이 저작권법과 관련이 있는지, 어떤 법률 개념·조문·판례 유형을 지향하는지 판별합니다. 신뢰도가 낮거나 비관련일 때는 조기 차단으로 비용과 오류를 줄입니다.

### 분석 항목

```python
{
    "is_related": bool,           # 저작권법 관련 여부
    "key_concepts": [str],        # 핵심 법률 개념
    "search_terms": [str],        # 검색 최적화 용어
    "question_type": str,         # 질문 유형
    "confidence": float           # 판단 신뢰도
}
```

### 프롬프트

프롬프트에 법률 도메인 지식을 전제로 한 JSON 형식 출력을 강제하여 후처리의 일관성을 확보합니다.

```python
prompt = f"""
당신은 저작권법 전문가입니다. 다음 질문을 분석하세요.

질문: {query}

분석 항목:
1. is_related: 저작권법 관련 여부 (true/false)
2. key_concepts: 핵심 법률 개념 추출 (예: 저작재산권, 공정이용)
3. search_terms: 검색에 유용한 용어 (예: 침해, 보호기간, 복제권)
4. question_type: 질문 유형 (예: 정의, 기준, 판례, 절차)
5. confidence: 판단 신뢰도 (0.0 ~ 1.0)

출력은 JSON 형식으로만 작성하세요.
"""
```

### 예시

**입력**: "음악을 샘플링해서 사용하면 저작권 침해인가요?"

**출력**:
```json
{
    "is_related": true,
    "key_concepts": ["샘플링", "저작재산권", "복제권", "저작권 침해"],
    "search_terms": ["음악 샘플링", "저작권 침해 기준", "이차적저작물", "실질적 유사성"],
    "question_type": "판단 기준",
    "confidence": 0.95
}
```

### 조기 차단 메커니즘

```python
def intent_analysis(query):
    """의도 분석"""
    intent = analyze_with_gpt4o(query)
    
    if not intent['is_related']:
        return {
            'stop': True,
            'message': "죄송합니다. 저작권법 관련 질문에만 답변할 수 있습니다."
        }
    
    if intent['confidence'] < 0.5:
        return {
            'stop': True,
            'message': "질문을 좀 더 명확히 해주시겠어요?"
        }
    
    return {
        'stop': False,
        'intent': intent
    }
```

---

## ↕️ 리랭킹 메커니즘

### 개요

1차 검색 결과를 질문 중심으로 다시 채점해 순서를 재구성하는 단계로, 다국어 Cross-Encoder를 사용해 법률 표현의 뉘앙스를 반영합니다.

### 모델 정보

- **모델명**: `jina-reranker-v2-base-multilingual`
- **입력**: 질문 + 문서 쌍
- **출력**: 관련도 점수 (0.0 ~ 1.0)
- **처리속도**: 평균 0.8초 (10개 문서 기준)

### 리랭킹 알고리즘

```python
def rerank(query, documents, top_k=5):
    """Jina AI 리랭킹"""
    
    # 1. API 호출
    response = requests.post(
        url="https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_k
        }
    )
    
    # 2. 점수 파싱
    results = response.json()['results']
    
    # 3. 재정렬
    reranked_documents = [
        {
            'document': documents[r['index']],
            'relevance_score': r['relevance_score']
        }
        for r in results
    ]
    
    # 4. 점수 순 정렬
    reranked_documents.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return [r['document'] for r in reranked_documents]
```

### 리랭킹 전후 비교

#### 벡터 검색 결과 (FAISS)

| 순위 | 문서 | 유사도 점수 |
|------|------|-------------|
| 1 | 저작권법 제2조 (정의) | 0.92 |
| 2 | 판례: 음악 샘플링 사건 | 0.89 |
| 3 | 저작재산권 개념 | 0.87 |
| 4 | 저작인격권 관련 | 0.85 |
| 5 | 저작권 등록 절차 | 0.83 |

#### 리랭킹 결과 (Jina AI)

| 순위 | 문서 | 관련도 점수 |
|------|------|-------------|
| 1 | 판례: 음악 샘플링 사건 | 0.95 |
| 2 | 저작권법 제2조 (정의) | 0.91 |
| 3 | 저작재산권 개념 | 0.88 |
| 4 | 저작인격권 관련 | 0.72 |
| 5 | 저작권 등록 절차 | 0.68 |

**개선 효과**: 질문과 가장 관련성 높은 판례가 1위로 상승

---

## 💬 답변 생성

### 개요

LLM을 사용하여 근거 문서를 토대로 결론→근거→설명→출처의 구조화된 답변을 제공합니다. 법령·판례는 가능한 한 조문 번호·사건번호를 명시하도록 하여 답변의 신뢰성을 높입니다. 

### 프롬프트 구조

```python
prompt = f"""
당신은 저작권법 전문 법률자문 AI입니다.

질문: {query}

참조 문서:
{format_documents(documents)}

답변 작성 규칙:
1. 질문에 직접적으로 답변
2. 관련 법령 조문 인용 (예: 저작권법 제X조)
3. 판례가 있다면 사건번호 명시
4. 명확하고 구체적으로 설명
5. 출처를 반드시 표시

답변:
"""
```

### 문서 포맷팅

```python
def format_documents(documents):
    """문서를 프롬프트용 텍스트로 변환"""
    formatted = []
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        content = doc.page_content
        
        # 메타데이터 포맷팅
        if metadata.get('doc_type') == 'case':
            header = f"[판례 {i}] {metadata['사건번호']} ({metadata['선고일자']})"
        else:
            header = f"[법령 {i}] {metadata['조문']}"
        
        formatted.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted)
```

### 생성 파라미터

```python
generation_config = {
    "model": "gpt-4-turbo",
    "temperature": 0.3,      # 낮은 온도 (일관성)
    "max_tokens": 1000,      # 최대 토큰
    "top_p": 0.9,            # Nucleus sampling
    "frequency_penalty": 0.3, # 반복 감소
    "presence_penalty": 0.1   # 다양성 증가
}
```

---

## 💾 데이터 스키마

스키마는 법률 도메인의 검증 가능한 필드(조문, 사건번호, 선고일) 를 중심으로 설계되어, 인용 정확도와 추적 가능성을 보장합니다.

### 판례 데이터

```python
{
    "사건번호": str,           # 예: "2019다123456"
    "법원": str,              # 예: "대법원"
    "선고일자": str,           # 예: "2020-01-15"
    "사건명": str,            # 예: "저작권침해금지 등"
    "판결유형": str,           # 예: "판결"
    "사건종류": str,           # 예: "민사"
    "판시사항": str,           # 판례의 핵심 쟁점
    "판결요지": str,           # 법원의 판단 요약
    "참조조문": [str],        # 예: ["저작권법 제2조", "저작권법 제10조"]
    "참조판례": [str],        # 관련 판례 사건번호
    "전문": str,              # 판결문 전체 텍스트
    "주문": str               # 판결 결과
}
```

### 법령 데이터

```python
{
    "법령명": str,            # 예: "저작권법"
    "조문": str,              # 예: "제2조"
    "항": int,               # 항 번호
    "호": int,               # 호 번호
    "조문제목": str,          # 예: "정의"
    "조문내용": str,          # 조문 전체 텍스트
    "시행일": str,            # 예: "2020-02-04"
    "개정이력": [str]        # 개정 날짜 목록
}
```

### 청크 메타데이터

```python
{
    "doc_id": str,            # 원본 문서 ID
    "doc_type": str,          # "case" 또는 "law"
    "chunk_index": int,       # 청크 순서
    "chunk_text": str,        # 청크 텍스트
    "source": str,            # 출처 (사건번호 또는 조문)
    "date": str,              # 선고일자 또는 시행일
    "section": str            # 문서 내 섹션 (예: "판시사항", "제2조")
}
```

### 벡터스토어 인덱스

```python
# FAISS Index
{
    "index": faiss.Index,     # FAISS 인덱스 객체
    "docstore": {             # 문서 저장소
        "doc_id": Document    # 문서 객체
    },
    "index_to_docstore_id": { # 인덱스-문서 매핑
        int: str
    }
}
```

---

## 🔢 알고리즘 상세

### 청킹 알고리즘

```python
def chunk_document(text, chunk_size=600, chunk_overlap=150):
    """
    RecursiveCharacterTextSplitter 알고리즘
    
    1. 구분자 우선순위에 따라 분할
    2. 청크 크기 초과 시 다음 구분자로 재분할
    3. 오버랩 적용하여 문맥 유지
    """
    
    # 구분자 우선순위
    separators = [
        "\n\n",    # 단락
        "\n",      # 줄
        ". ",      # 문장
        ", ",      # 절
        " ",       # 단어
        ""         # 문자
    ]
    
    chunks = []
    current_chunk = ""
    
    for separator in separators:
        splits = text.split(separator)
        
        for split in splits:
            if len(current_chunk) + len(split) <= chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # 오버랩 적용
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + split + separator
                else:
                    current_chunk = split + separator
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```

### 유사도 검색

```python
def similarity_search(query, k=10):
    """
    FAISS 코사인 유사도 검색
    
    1. 쿼리 임베딩 생성
    2. FAISS 인덱스에서 k-NN 검색
    3. 유사도 점수 계산
    4. 메타데이터 결합
    """
    
    # 1. 쿼리 임베딩
    query_embedding = embedding_model.embed_query(query)
    
    # 2. FAISS 검색
    distances, indices = index.search(
        np.array([query_embedding]), 
        k
    )
    
    # 3. 문서 검색
    documents = []
    for i, idx in enumerate(indices[0]):
        doc_id = index_to_docstore_id[idx]
        doc = docstore[doc_id]
        
        # 유사도 점수 (거리 → 유사도 변환)
        similarity = 1 - (distances[0][i] / 2)
        
        documents.append({
            'document': doc,
            'similarity': similarity
        })
    
    return documents
```

### RAGAS 평가 알고리즘

```python
def evaluate_answer(query, answer, contexts):
    """
    RAGAS 평가 메트릭
    
    1. Faithfulness: 답변의 충실성
    2. Relevancy: 답변의 관련성
    """
    
    # 1. Faithfulness (충실성)
    # - 답변의 각 문장이 컨텍스트에서 지지되는지 확인
    faithfulness_score = calculate_faithfulness(answer, contexts)
    
    # 2. Relevancy (관련성)
    # - 답변이 질문과 얼마나 관련있는지 평가
    relevancy_score = calculate_relevancy(query, answer)
    
    # 3. 종합 점수
    combined_score = (
        faithfulness_score * 0.5 +
        relevancy_score * 0.5
    )
    
    return {
        'faithfulness': faithfulness_score,
        'relevancy': relevancy_score,
        'combined': combined_score
    }

def calculate_faithfulness(answer, contexts):
    """충실성 계산"""
    # LLM으로 각 문장의 지지 여부 판단
    sentences = split_sentences(answer)
    supported_count = 0
    
    for sentence in sentences:
        is_supported = check_support(sentence, contexts)
        if is_supported:
            supported_count += 1
    
    return supported_count / len(sentences)

def calculate_relevancy(query, answer):
    """관련성 계산"""
    # LLM으로 답변과 질문의 관련성 판단
    prompt = f"""
    질문: {query}
    답변: {answer}
    
    위 답변이 질문에 얼마나 관련있는지 0.0 ~ 1.0 점수로 평가하세요.
    """
    
    score = llm_evaluate(prompt)
    return score
```

---

## 📊 성능 최적화

### 배치 처리

```python
def batch_embed(texts, batch_size=500):
    """임베딩 배치 처리"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

### 캐싱

```python
@lru_cache(maxsize=1000)
def cached_search(query, k):
    """검색 결과 캐싱"""
    return vectorstore.similarity_search(query, k=k)
```

### 병렬 처리

```python
def parallel_multi_query_search(queries, k):
    """멀티쿼리 병렬 검색"""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(vectorstore.similarity_search, q, k)
            for q in queries
        ]
        results = [f.result() for f in futures]
    
    return merge_results(results)
```

---

## 🛡️ 품질 보증

데이터·모델·파이프라인 전 단계에 적용되는 예방적 통제입니다. 최소 길이, 한글 비율, 비문자 판별 등 기초 검증으로 저품질 청크를 차단합니다.

### 데이터 검증

```python
def validate_chunk_quality(chunk):
    """청크 품질 검증"""
    checks = {
        'min_length': len(chunk) >= 100,
        'max_length': len(chunk) <= 1000,
        'has_content': len(chunk.strip()) > 0,
        'not_only_punctuation': any(c.isalnum() for c in chunk),
        'korean_ratio': sum(is_korean(c) for c in chunk) / len(chunk) > 0.5
    }
    
    return all(checks.values())
```

### 평가 신뢰도

```python
def confidence_check(evaluation):
    """평가 신뢰도 확인"""
    # 점수 간 차이가 클 때 재평가
    if abs(evaluation.faithfulness - evaluation.relevancy) > 0.3:
        return False
    
    # 극단적 점수는 재평가
    if evaluation.faithfulness < 0.1 or evaluation.faithfulness > 0.95:
        return False
    
    return True
```

---

## 📈 메트릭 모니터링

요청별 타이밍·비용·점수·k 값 등 운영 지표를 수집해 병목과 비용 급증을 조기 경보합니다. 핵심 지표는 대시보드로 상시 관리합니다.

### 로깅 데이터

```python
log_data = {
    "timestamp": datetime.now(),
    "query": query,
    "retry_count": retry_count,
    "retrieved_docs": len(documents),
    "reranked_docs": len(reranked_documents),
    "evaluation": {
        "faithfulness": evaluation.faithfulness,
        "relevancy": evaluation.relevancy,
        "passing": is_passing,
        "fallback": is_fallback
    },
    "timing": {
        "intent_analysis": 0.5,
        "retrieval": 1.2,
        "reranking": 0.8,
        "generation": 2.5,
        "evaluation": 1.0,
        "total": 6.0
    },
    "costs": {
        "openai": 0.025,
        "jina": 0.003,
        "total": 0.028
    }
}
```

---

## 🎓 추가 리소스

- **[README.md](./README.md)** - 프로젝트 전체 개요
- **[README_ENV.md](./README_ENV.md)** - 환경변수 설정 가이드

---

<div align="center">

**[⬆ 맨 위로](#모델-아키텍처-및-데이터-명세)**

</div>
