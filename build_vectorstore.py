# -*- coding: utf-8 -*-
"""
로컬 벡터스토어 빌드 스크립트

사용법:
    python build_vectorstore.py --case 판례파일.xlsx --law 법령파일.xlsx

필수 환경변수 (.env):
    OPENAI_API_KEY=sk-...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from data_pipeline.vectorstore import (
    create_vectorstore_in_batches,
    save_vectorstore,
)
from utils.logger import log


# ─────────────────────────────────────────────
# 판례 Document 생성
# ─────────────────────────────────────────────
def build_case_docs(df: pd.DataFrame) -> list[Document]:
    separators = [
        "\n\n", ".\n", ". ",
        "【이 유】", "【주 문】",
        "\n1. ", "\n2. ", "\n3. ",
        "\n가. ", "\n나. ", "\n다. ", "\n라. ",
        ") ", ", ", " "
    ]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=settings.chunking.case_chunk_size,       # 기본 600
        chunk_overlap=settings.chunking.case_chunk_overlap, # 기본 150
        length_function=len,
        is_separator_regex=False,
    )

    docs = []
    skipped = 0
    for idx, row in df.iterrows():
        parts = []
        if str(row.get("판시사항", "")).strip():
            parts.append(f"【판시사항】\n{str(row['판시사항']).strip()}")
        if str(row.get("판결요지", "")).strip():
            parts.append(f"【판결요지】\n{str(row['판결요지']).strip()}")
        if str(row.get("판례내용", "")).strip():
            parts.append(f"【판례내용】\n{str(row['판례내용']).strip()}")

        combined = "\n\n".join(parts)
        if not combined.strip():
            skipped += 1
            continue

        metadata = {
            "사건명":     str(row.get("사건명", "")).strip(),
            "사건번호":   str(row.get("사건번호", "")).strip(),
            "선고일자":   str(row.get("선고일자", "")).strip(),
            "법원명":     str(row.get("법원명", "")).strip(),
            "사건종류명": str(row.get("사건종류명", "")).strip(),
            "판결유형":   str(row.get("판결유형", "")).strip(),
            "참조조문":   str(row.get("참조조문", "")).strip(),
            "doc_id":     f"case_{idx+1}",
            "문서유형":   "판례",
        }

        for chunk in splitter.split_text(combined):
            docs.append(Document(page_content=chunk, metadata=metadata))

    log.info(f"판례 청킹 완료: {len(docs)}개 청크 (건너뜀: {skipped}개 행)")
    return docs


# ─────────────────────────────────────────────
# 법령 Document 생성 (계층형 구조 재조합)
# 구조: 조문번호 > 항번호 > 호번호 > 목번호
# 시행령/부칙은 별도 컬럼(시행령_조문내용, 부칙_조문내용) 사용
# ─────────────────────────────────────────────
def _is_empty(val) -> bool:
    """NaN 또는 빈 문자열 여부 확인"""
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def _reconstruct_article(group: pd.DataFrame) -> str:
    """
    하나의 조문(조문번호 기준 그룹)을 계층 구조로 재조합.
    항/호/목이 있으면 들여쓰기로 구분.
    """
    lines = []

    # 조문 제목 + 본문 (조문여부 == '조문' 인 행의 항내용이 본문)
    for _, row in group.iterrows():
        # 조문 본문 (항번호 없는 행 = 조문 자체 내용)
        if _is_empty(row.get("항번호")):
            if not _is_empty(row.get("항내용")):
                lines.append(str(row["항내용"]).strip())
            continue

        # 항
        hang = f"  {str(row['항번호']).strip()}" if not _is_empty(row.get("항번호")) else ""
        hang_content = str(row.get("항내용", "")).strip() if not _is_empty(row.get("항내용")) else ""
        ho = f"    {str(row['호번호']).strip()}" if not _is_empty(row.get("호번호")) else ""
        ho_content = str(row.get("호내용", "")).strip() if not _is_empty(row.get("호내용")) else ""
        mok = f"      {str(row['목번호']).strip()}" if not _is_empty(row.get("목번호")) else ""
        mok_content = str(row.get("목내용", "")).strip() if not _is_empty(row.get("목내용")) else ""

        if mok and mok_content:
            lines.append(f"{mok} {mok_content}")
        elif ho and ho_content:
            lines.append(f"{ho} {ho_content}")
        elif hang and hang_content:
            lines.append(f"{hang} {hang_content}")

    return "\n".join(lines).strip()


def build_law_docs(df: pd.DataFrame, doc_type: str = "법령") -> list[Document]:
    """
    법령/시행령/부칙 DataFrame → Document 리스트 생성

    실제 파일 구조:
        법령:   조문번호, 조문제목, 조문여부, 항번호, 항내용, 호번호, 호내용, 목번호, 목내용 (계층형)
        시행령: 시행령_조문번호, 시행령_조문제목, 시행령_조문내용 (단일 컬럼)
        부칙:   조문번호, 조문제목, 조문내용, 공포일자, 공포번호 (단일 컬럼)
    """
    separators = ["\n   ", "\n     ", "\n", ". ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=settings.chunking.law_chunk_size,
        chunk_overlap=settings.chunking.law_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    docs = []
    skipped = 0
    doc_counter = 0

    # ── 시행령: 시행령_조문내용 단일 컬럼 ──
    if doc_type == "시행령":
        for idx, row in df.iterrows():
            content = str(row.get("시행령_조문내용", "")).strip()
            if _is_empty(content):
                skipped += 1
                continue
            metadata = {
                "조문번호": str(row.get("시행령_조문번호", "")).strip(),
                "조문제목": str(row.get("시행령_조문제목", "")).strip(),
                "doc_id":   f"시행령_{idx+1}",
                "문서유형": "시행령",
            }
            for chunk in splitter.split_text(content):
                docs.append(Document(page_content=chunk, metadata=metadata))
                doc_counter += 1

    # ── 부칙: 조문내용 단일 컬럼 ──
    elif doc_type == "부칙":
        for idx, row in df.iterrows():
            content = str(row.get("조문내용", "")).strip()
            if _is_empty(content):
                skipped += 1
                continue
            metadata = {
                "조문번호": str(row.get("조문번호", "")).strip(),
                "조문제목": str(row.get("조문제목", "")).strip(),
                "공포일자": str(row.get("공포일자", "")).strip(),
                "공포번호": str(row.get("공포번호", "")).strip(),
                "doc_id":   f"부칙_{idx+1}",
                "문서유형": "부칙",
            }
            for chunk in splitter.split_text(content):
                docs.append(Document(page_content=chunk, metadata=metadata))
                doc_counter += 1

    # ── 법령 본문: 계층형 구조 재조합 ──
    else:
        grouped = df.groupby("조문번호", sort=False)
        for article_no, group in grouped:
            # 전문(前文) 등 특수 행 처리
            if str(article_no).strip() in ("0", "00", "전문", "1조"):
                first_row = group.iloc[0]
                if str(first_row.get("조문여부", "")).strip() == "전문":
                    for _, row in group.iterrows():
                        content = str(row.get("항내용", "")).strip()
                        if _is_empty(content):
                            continue
                        metadata = {
                            "조문번호": str(article_no),
                            "조문제목": str(row.get("조문제목", "")).strip(),
                            "doc_id":   f"law_{doc_counter+1}",
                            "문서유형": "법령",
                        }
                        docs.append(Document(page_content=content, metadata=metadata))
                        doc_counter += 1
                    continue

            content = _reconstruct_article(group)
            if not content:
                skipped += 1
                continue

            first_row = group.iloc[0]
            metadata = {
                "조문번호": str(article_no).strip(),
                "조문제목": str(first_row.get("조문제목", "")).strip(),
                "조문여부": str(first_row.get("조문여부", "")).strip(),
                "doc_id":   f"law_{doc_counter+1}",
                "문서유형": "법령",
            }
            for chunk in splitter.split_text(content):
                docs.append(Document(page_content=chunk, metadata=metadata))
                doc_counter += 1

    log.info(f"{doc_type} 청킹 완료: {len(docs)}개 청크 (건너뜀: {skipped}개)")
    return docs


# ─────────────────────────────────────────────
# 해석례 Document 생성
# 컬럼: 안건명, 안건번호, 질의기관명, 회신일자,
#       질의요지, 질의요지_요약, 회답 및 이유, 관계법령
# ─────────────────────────────────────────────
def build_interpretation_docs(df: pd.DataFrame) -> list[Document]:
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=settings.chunking.law_chunk_size,
        chunk_overlap=settings.chunking.law_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    docs = []
    skipped = 0
    for idx, row in df.iterrows():
        # 질의요지 + 회답 및 이유를 합쳐서 본문으로 사용
        parts = []
        if not _is_empty(row.get("질의요지_요약")):
            parts.append(f"【질의요지】\n{str(row['질의요지_요약']).strip()}")
        elif not _is_empty(row.get("질의요지")):
            parts.append(f"【질의요지】\n{str(row['질의요지']).strip()}")
        if not _is_empty(row.get("회답 및 이유")):
            parts.append(f"【회답 및 이유】\n{str(row['회답 및 이유']).strip()}")

        content = "\n\n".join(parts)
        if not content.strip():
            skipped += 1
            continue

        metadata = {
            "안건명":    str(row.get("안건명", "")).strip()[:100],
            "안건번호":  str(row.get("안건번호", "")).strip(),
            "질의기관":  str(row.get("질의기관명", "")).strip(),
            "회신일자":  str(row.get("회신일자", "")).strip(),
            "관계법령":  str(row.get("관계법령", "")).strip(),
            "doc_id":    f"interp_{idx+1}",
            "문서유형":  "해석례",
        }
        for chunk in splitter.split_text(content):
            docs.append(Document(page_content=chunk, metadata=metadata))

    log.info(f"해석례 청킹 완료: {len(docs)}개 청크 (건너뜀: {skipped}개)")
    return docs


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="저작권법 RAG 벡터스토어 빌드")
    parser.add_argument("--case",   required=True,  help="판례 엑셀 파일 경로 (.xlsx)")
    parser.add_argument("--law",    required=False, help="법령 본문 엑셀 파일 경로 (.xlsx)")
    parser.add_argument("--decree", required=False, help="시행령 엑셀 파일 경로 (.xlsx)")
    parser.add_argument("--suppl",  required=False, help="부칙 엑셀 파일 경로 (.xlsx)")
    parser.add_argument("--interp", required=False, help="해석례 엑셀 파일 경로 (.xlsx)")
    parser.add_argument("--output", default=None,   help="저장 경로 (기본: data/vectorstore)")
    parser.add_argument("--batch",  type=int, default=500, help="배치 크기 (기본: 500)")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("벡터스토어 빌드 시작")
    log.info(f"임베딩 모델: {settings.openai.embedding_model}")
    log.info("=" * 50)

    all_docs = []

    # 판례 로드
    case_path = Path(args.case)
    if not case_path.exists():
        log.error(f"판례 파일을 찾을 수 없습니다: {case_path}")
        sys.exit(1)
    log.info(f"판례 파일 로드: {case_path}")
    case_df = pd.read_excel(case_path)
    if "참조조문" in case_df.columns:
        case_df["참조조문"] = case_df["참조조문"].str.replace(r'\([^)]*\)', '', regex=True)
    log.info(f"판례 데이터: {len(case_df)}행")
    all_docs.extend(build_case_docs(case_df))

    # 법령 / 시행령 / 부칙 로드 (선택)
    for flag, label in [(args.law, "법령"), (args.decree, "시행령"), (args.suppl, "부칙")]:
        if not flag:
            continue
        p = Path(flag)
        if not p.exists():
            log.error(f"{label} 파일을 찾을 수 없습니다: {p}")
            sys.exit(1)
        log.info(f"{label} 파일 로드: {p}")
        df = pd.read_excel(p)
        log.info(f"{label} 데이터: {len(df)}행")
        all_docs.extend(build_law_docs(df, doc_type=label))

    # 해석례 로드 (선택)
    if args.interp:
        p = Path(args.interp)
        if not p.exists():
            log.error(f"해석례 파일을 찾을 수 없습니다: {p}")
            sys.exit(1)
        log.info(f"해석례 파일 로드: {p}")
        df = pd.read_excel(p)
        log.info(f"해석례 데이터: {len(df)}행")
        all_docs.extend(build_interpretation_docs(df))

    log.info(f"총 청크 수: {len(all_docs)}개")
    log.info("임베딩 + FAISS 인덱스 생성 중... (시간이 걸릴 수 있습니다)")

    vectorstore = create_vectorstore_in_batches(all_docs, batch_size=args.batch)
    save_vectorstore(vectorstore, save_path=args.output)

    log.info("=" * 50)
    log.info("빌드 완료!")
    log.info(f"저장 경로: {args.output or settings.paths.vectorstore_path}")
    log.info(f"총 벡터 수: {vectorstore.index.ntotal}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()