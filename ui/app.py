# -*- coding: utf-8 -*-
"""
Streamlit UI 애플리케이션
법률자문 AI 웹 인터페이스
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from datetime import datetime


from core.pipeline import LegalQAChain
from utils import log


# 페이지 설정
st.set_page_config(
    page_title="저작권법 법률자문 AI",
    page_icon="⚖️",
    layout="wide"
)

# 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_qa_chain():
    """QA 체인 로드 (캐싱)"""
    return LegalQAChain()


def format_source_documents(docs):
    """참조 문서 포맷"""
    formatted_docs = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        doc_type = metadata.get('문서유형', '알 수 없음')
        
        if doc_type == '법령':
            article_num = metadata.get('조문번호', '')
            article_title = metadata.get('조문제목', '')
            formatted_docs.append({
                "번호": i,
                "유형": "법령",
                "제목": f"저작권법 제{article_num} {article_title}",
                "내용": doc.page_content[:200] + "..."
            })
        
        elif doc_type == '판례':
            case_num = metadata.get('사건번호', '')
            case_name = metadata.get('사건명', '')
            court = metadata.get('법원명', '')
            formatted_docs.append({
                "번호": i,
                "유형": "판례",
                "제목": f"{court} {case_num}",
                "사건명": case_name,
                "내용": doc.page_content[:200] + "..."
            })
        
        else:
            formatted_docs.append({
                "번호": i,
                "유형": doc_type,
                "제목": "기타 참조",
                "내용": doc.page_content[:200] + "..."
            })
    
    return formatted_docs


def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown('<p class="main-header">⚖️ 저작권법 법률자문 AI</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    이 AI는 **저작권법 법령, 판례, 해석례**를 기반으로 법률 정보를 제공합니다.
    
    ⚠️ **주의사항**: 본 서비스는 법적 조언이 아닌 정보 제공 목적입니다.
    """)
    
    # QA 체인 로드
    try:
        qa_chain = load_qa_chain()
        log.info("QA 체인 로드 완료")
    except Exception as e:
        st.error(f"시스템 초기화 오류: {str(e)}")
        st.stop()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 참조 문서 표시
            if "sources" in message and message["sources"]:
                with st.expander("📚 참조 문서"):
                    for doc in message["sources"]:
                        st.markdown(f"**[{doc['유형']} {doc['번호']}] {doc['제목']}**")
                        if doc['유형'] == '판례' and 'case_name' in doc:
                            st.caption(f"사건명: {doc['사건명']}")
                        st.text(doc['내용'])
                        st.divider()
    
    # 사용자 입력
    if prompt := st.chat_input("저작권법에 대해 궁금한 것을 물어보세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    # QA 체인 실행
                    result = qa_chain.run(prompt)
                    
                    # 답변 표시
                    st.markdown(result["answer"])
                    
                    # 참조 문서 표시
                    if result["source_documents"]:
                        formatted_sources = format_source_documents(
                            result["source_documents"]
                        )
                        
                        with st.expander("📚 참조 문서"):
                            for doc in formatted_sources:
                                st.markdown(f"**[{doc['유형']} {doc['번호']}] {doc['제목']}**")
                                if doc['유형'] == '판례' and '사건명' in doc:
                                    st.caption(f"사건명: {doc['사건명']}")
                                st.text(doc['내용'])
                                st.divider()
                        
                        # 메시지에 참조 문서 추가
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": formatted_sources
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })
                    
                    log.info(f"질의응답 완료: {prompt[:50]}")
                
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    log.error(f"질의응답 오류: {str(e)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # 사이드바
    with st.sidebar:
        st.header("ℹ️ 정보")
        
        st.markdown("""
        ### 사용 가능한 데이터
        - 📜 저작권법 법령
        - ⚖️ 저작권법 판례
        - 📋 저작권법 해석례
        
        ### 사용 팁
        1. 구체적으로 질문하세요
        2. 조문이나 판례를 직접 언급할 수 있습니다
        3. 참조 문서를 확인하세요
        """)
        
        st.divider()
        
        # 대화 초기화 버튼
        if st.button("🔄 대화 초기화"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
