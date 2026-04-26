# -*- coding: utf-8 -*-
"""
배치 평가 스크립트
Q&A 데이터를 활용한 RAG 파이프라인 성능 평가

사용법:
    python evaluate_batch.py \
        --qa_file data/qa/Q_A데이터_통합.xlsx \
        --n_samples 20 \
        --output results/evaluation_results.xlsx
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from config.settings import settings
from data_pipeline.vectorstore import load_vectorstore
from core.pipeline import LegalRAGPipeline
from utils.logger import log


def load_qa_samples(qa_path: str, n_samples: int, random_seed: int = 42) -> pd.DataFrame:
    """Q&A 데이터 로드 및 샘플링"""
    df = pd.read_excel(qa_path)

    required_cols = ["question", "answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df.dropna(subset=required_cols)

    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

    log.info(f"Q&A 샘플 로드 완료: {len(df)}개")
    return df


def run_pipeline_on_samples(
    pipeline: LegalRAGPipeline,
    qa_df: pd.DataFrame,
    checkpoint_path: str = "results/checkpoint.json"
) -> list[dict]:
    """각 질문을 파이프라인에 통과시켜 결과 수집 (체크포인트 지원)"""
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # 기존 체크포인트 로드
    results = []
    start_idx = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results)
            log.info(f"체크포인트 로드 완료: {start_idx}개 결과 복원, {len(qa_df) - start_idx}개 남음")
        except Exception as e:
            log.warning(f"체크포인트 로드 실패, 처음부터 시작: {str(e)}")
            results = []
            start_idx = 0

    remaining_df = qa_df.iloc[start_idx:].reset_index(drop=True)

    for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="파이프라인 실행"):
        question = str(row["question"]).strip()
        ground_truth = str(row["answer"]).strip()

        try:
            result = pipeline.process(question)
            contexts = [doc.page_content for doc in result.reranked_documents]

            results.append({
                "question":     question,
                "answer":       result.answer,
                "ground_truth": ground_truth,
                "contexts":     contexts,
                "retry_count":  result.retry_count,
                "is_fallback":  result.is_fallback_used,
                "faithfulness": result.evaluation.faithfulness_score,
                "relevancy":    result.evaluation.relevancy_score,
                "passed":       result.evaluation.passed,
                "error":        None,
            })

        except Exception as e:
            log.exception(f"[{start_idx + idx}] 파이프라인 오류: {str(e)}")
            results.append({
                "question":     question,
                "answer":       "",
                "ground_truth": ground_truth,
                "contexts":     [],
                "retry_count":  0,
                "is_fallback":  False,
                "faithfulness": 0.0,
                "relevancy":    0.0,
                "passed":       False,
                "error":        str(e),
            })

        # 매 질문마다 체크포인트 저장
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 완료 후 체크포인트 삭제
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        log.info("체크포인트 삭제 완료")

    return results


def run_ragas_evaluation(results: list[dict]) -> dict:
    valid = [r for r in results if not r["error"] and r["contexts"]]
    log.info(f"RAGAS 평가 대상: {len(valid)}개")

    if not valid:
        log.error("평가 가능한 결과가 없습니다")
        return {}

    # 5개씩 나눠서 평가 (TPM 한도 초과 방지)
    import time
    all_faith, all_rel, all_cp = [], [], []
    batch_size = 5

    base_llm = ChatOpenAI(
        model_name=settings.openai.evaluation_model,
        temperature=0,
        openai_api_key=settings.openai.api_key,
        max_tokens=settings.openai.max_tokens_evaluation,
        request_timeout=60
    )
    evaluator_llm = LangchainLLMWrapper(base_llm)

    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        log.info(f"RAGAS 평가 배치 {i//batch_size + 1}: {len(batch)}개")

        data = {
            "question":     [r["question"]     for r in batch],
            "answer":       [r["answer"]       for r in batch],
            "contexts":     [r["contexts"]     for r in batch],
            "ground_truth": [r["ground_truth"] for r in batch],
        }
        dataset = Dataset.from_dict(data)

        ragas_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=evaluator_llm
        )

        all_faith.append(float(ragas_result["faithfulness"]))
        all_rel.append(float(ragas_result["answer_relevancy"]))
        all_cp.append(float(ragas_result["context_precision"]))

        if i + batch_size < len(valid):
            log.info("Rate limit 방지: 30초 대기 중...")
            time.sleep(30)

    return {
        "faithfulness":      sum(all_faith) / len(all_faith),
        "answer_relevancy":  sum(all_rel)   / len(all_rel),
        "context_precision": sum(all_cp)    / len(all_cp),
    }


def save_results(
    results: list[dict],
    ragas_scores: dict,
    output_path: str
):
    """결과 저장 (엑셀)"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 개별 결과 시트
    detail_df = pd.DataFrame([{
        "질문":        r["question"],
        "생성 답변":   r["answer"],
        "정답":        r["ground_truth"],
        "재시도 횟수": r["retry_count"],
        "Fallback":    r["is_fallback"],
        "Faithfulness": r["faithfulness"],
        "Relevancy":   r["relevancy"],
        "통과":        r["passed"],
        "오류":        r["error"] or "",
    } for r in results])

    # 요약 시트
    passed_count = sum(1 for r in results if r["passed"])
    error_count  = sum(1 for r in results if r["error"])
    summary_data = {
        "항목": [
            "평가 일시",
            "총 샘플 수",
            "통과 수",
            "통과율",
            "오류 수",
            "RAGAS Faithfulness",
            "RAGAS Answer Relevancy",
            "RAGAS Context Precision",
        ],
        "값": [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            len(results),
            passed_count,
            f"{passed_count / len(results) * 100:.1f}%",
            error_count,
            f"{ragas_scores.get('faithfulness', 'N/A'):.4f}" if ragas_scores else "N/A",
            f"{ragas_scores.get('answer_relevancy', 'N/A'):.4f}" if ragas_scores else "N/A",
            f"{ragas_scores.get('context_precision', 'N/A'):.4f}" if ragas_scores else "N/A",
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="요약", index=False)
        detail_df.to_excel(writer, sheet_name="상세결과", index=False)

    log.info(f"결과 저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG 파이프라인 배치 평가")
    parser.add_argument("--qa_file",   required=True,          help="Q&A 엑셀 파일 경로")
    parser.add_argument("--n_samples", type=int, default=20,   help="샘플 수 (기본: 20)")
    parser.add_argument("--seed",      type=int, default=42,   help="랜덤 시드 (기본: 42)")
    parser.add_argument("--output",    default="results/evaluation_results.xlsx",
                        help="결과 저장 경로")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("배치 평가 시작")
    log.info(f"Q&A 파일: {args.qa_file}")
    log.info(f"샘플 수: {args.n_samples}")
    log.info("=" * 60)

    # 1. Q&A 샘플 로드
    qa_df = load_qa_samples(args.qa_file, args.n_samples, args.seed)

    # 2. 파이프라인 초기화
    log.info("벡터스토어 로드 중...")
    vectorstore = load_vectorstore()
    pipeline = LegalRAGPipeline(vectorstore)

    # 3. 파이프라인 실행 (기존 결과 있으면 재사용)
    pipeline_results_path = Path(args.output).parent / "pipeline_results.json"

    if pipeline_results_path.exists():
        log.info("기존 파이프라인 결과 로드, RAGAS 평가만 실행")
        with open(pipeline_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        log.info(f"파이프라인 결과 로드 완료: {len(results)}개")
    else:
        results = run_pipeline_on_samples(pipeline, qa_df)
        pipeline_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pipeline_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"파이프라인 결과 저장: {pipeline_results_path}")

    # 4. RAGAS 평가
    ragas_scores = run_ragas_evaluation(results)

    # 5. 결과 출력
    log.info("=" * 60)
    log.info("평가 결과 요약")
    log.info("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    log.info(f"총 샘플:     {len(results)}개")
    log.info(f"통과:        {passed}개 ({passed/len(results)*100:.1f}%)")
    if ragas_scores:
        log.info(f"Faithfulness:      {ragas_scores['faithfulness']:.4f}")
        log.info(f"Answer Relevancy:  {ragas_scores['answer_relevancy']:.4f}")
        log.info(f"Context Precision: {ragas_scores['context_precision']:.4f}")

    # 6. 결과 저장
    save_results(results, ragas_scores, args.output)

    log.info("배치 평가 완료")


if __name__ == "__main__":
    main()