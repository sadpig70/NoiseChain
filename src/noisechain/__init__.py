"""
NoiseChain - Physical Trust Verification Network

물리적 환경 노이즈 기반 신뢰 검증 네트워크.
Proof-of-Experience (PoX) 토큰을 통해 "특정 장비가 특정 시간·환경을 
실제로 경험했다"를 증명합니다.

주요 모듈:
    - models: 데이터 모델 (SensorType, Sample, TimeSeries)
    - sensors: 가상 센서 드라이버
    - pipeline: 특징 추출 및 상관 서명
    - token: PoXToken 스키마 및 조립
    - storage: SQLite 저장소
    - verification: 검증 엔진

사용 예시:
    >>> from noisechain import NoiseChainPipeline
    >>> pipeline = NoiseChainPipeline()
    >>> result = pipeline.generate_and_store()
    >>> print(result.success)
"""

__version__ = "0.1.0"
__author__ = "NoiseChain Team"

# 버전 정보
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "alpha",
}

# E2E 파이프라인 export
from noisechain.engine import (
    NoiseChainPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline,
)

__all__ = [
    "NoiseChainPipeline",
    "PipelineConfig",
    "PipelineResult",
    "create_pipeline",
    "__version__",
]

# 패키지 초기화 시 로깅 설정
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

