# -*- coding: utf-8 -*-
"""
메인 실행 파일
Streamlit 애플리케이션 실행
"""
import sys
import subprocess
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Streamlit 애플리케이션 실행"""
    streamlit_app = Path(__file__).parent / "ui" / "app.py"
    
    if not streamlit_app.exists():
        print(f"오류: {streamlit_app} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    subprocess.run([
        "streamlit", "run",
        str(streamlit_app),
        "--server.port", "8501",
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
