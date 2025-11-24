# -*- coding: utf-8 -*-
"""
로깅 유틸리티
loguru 기반 구조화된 로깅
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logger():
    """로거 설정"""
    
    # 기본 로거 제거
    logger.remove()
    
    # 콘솔 출력 설정
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # 파일 출력 설정 (일별 로테이션)
    log_file = settings.paths.log_dir / "app_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        rotation="00:00",  # 자정에 새 파일
        retention="30 days",  # 30일 보관
        compression="zip",  # 압축
        encoding="utf-8"
    )
    
    # 에러 로그 별도 파일
    error_log_file = settings.paths.log_dir / "error_{time:YYYY-MM-DD}.log"
    logger.add(
        str(error_log_file),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} | {message}\n{exception}",
        rotation="00:00",
        retention="90 days",  # 에러 로그는 90일 보관
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info("로거 설정 완료")
    return logger


# 전역 로거 인스턴스
log = setup_logger()
