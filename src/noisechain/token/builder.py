"""
NoiseChain 토큰 빌더

센서 데이터와 상관 서명에서 PoXToken을 생성하는 빌더 패턴 구현.
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np

from noisechain.models.noise_data import SensorType
from noisechain.pipeline.correlation import CorrelationResult, CorrelationSignature
from noisechain.pipeline.features import FeatureExtractor, FeatureVector
from noisechain.token.schema import (
    NoiseFingerprint,
    PoXToken,
    TokenType,
    TokenVersion,
)


@dataclass
class RiskConfig:
    """위험 점수 계산 설정"""
    entropy_weight: float = 0.3      # 엔트로피 가중치
    variance_weight: float = 0.3     # 분산 가중치
    correlation_weight: float = 0.4  # 상관 가중치
    low_entropy_threshold: float = 0.3  # 낮은 엔트로피 임계값
    high_variance_threshold: float = 0.8  # 높은 분산 임계값


class RiskScorer:
    """
    위험 점수 계산기
    
    노이즈 특성을 분석하여 위험 점수(0-100)를 계산합니다.
    높은 점수 = 높은 위험 (조작 가능성).
    
    계산 요소:
    - 엔트로피 부족: 랜덤성 부족 → 예측 가능
    - 분산 부족: 변화 부족 → 고정 패턴
    - 상관 이상: 센서 간 비정상 상관
    """
    
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
    
    def compute(
        self, 
        features: dict[str, FeatureVector],
        correlation: CorrelationResult
    ) -> float:
        """
        위험 점수 계산
        
        Args:
            features: 센서별 특징 벡터
            correlation: 상관 서명 결과
        
        Returns:
            위험 점수 (0.0 ~ 100.0)
        """
        entropy_risk = self._compute_entropy_risk(features)
        variance_risk = self._compute_variance_risk(features)
        correlation_risk = self._compute_correlation_risk(correlation)
        
        # 가중 합산
        score = (
            self.config.entropy_weight * entropy_risk +
            self.config.variance_weight * variance_risk +
            self.config.correlation_weight * correlation_risk
        )
        
        return min(100.0, max(0.0, score))
    
    def _compute_entropy_risk(self, features: dict[str, FeatureVector]) -> float:
        """엔트로피 기반 위험 (낮은 엔트로피 = 높은 위험)"""
        from noisechain.pipeline.features import FeatureType
        
        if not features:
            return 50.0
        
        entropies = []
        for fv in features.values():
            entropy = fv.get(FeatureType.ENTROPY, 0)
            entropies.append(entropy)
        
        if not entropies:
            return 50.0
        
        avg_entropy = np.mean(entropies)
        
        # 엔트로피 정규화 (0~10 범위 가정)
        normalized = min(1.0, avg_entropy / 10.0)
        
        # 낮은 엔트로피 = 높은 위험
        return (1.0 - normalized) * 100
    
    def _compute_variance_risk(self, features: dict[str, FeatureVector]) -> float:
        """분산 기반 위험 (낮은 분산 = 높은 위험)"""
        from noisechain.pipeline.features import FeatureType
        
        if not features:
            return 50.0
        
        variances = []
        for fv in features.values():
            var = fv.get(FeatureType.VARIANCE, 0)
            variances.append(var)
        
        if not variances:
            return 50.0
        
        avg_var = np.mean(variances)
        
        # 분산 정규화 (로그 스케일)
        normalized = min(1.0, np.log1p(avg_var) / 10.0)
        
        # 낮은 분산 = 높은 위험
        return (1.0 - normalized) * 100
    
    def _compute_correlation_risk(self, correlation: CorrelationResult) -> float:
        """상관 기반 위험 (비정상 패턴 = 높은 위험)"""
        if correlation.sensor_count < 2:
            return 0.0  # 단일 센서는 상관 위험 없음
        
        # 특징 벡터의 불균형 검사
        fv = correlation.feature_vector
        
        # 모든 값이 비슷하면 의심스러움
        if np.std(fv) < 0.01:
            return 50.0
        
        # 극단값 비율
        extreme = np.sum(np.abs(fv - 0.5) > 0.4) / len(fv)
        
        return extreme * 100


class TokenBuilder:
    """
    PoXToken 빌더
    
    센서 데이터에서 완전한 PoXToken을 생성합니다.
    
    Example:
        >>> builder = TokenBuilder(node_id=uuid.uuid4().bytes)
        >>> token = builder.from_sensor_data(sensor_data)
    """
    
    def __init__(
        self, 
        node_id: bytes | None = None,
        version: TokenVersion = TokenVersion.V1_MVP,
        token_type: TokenType = TokenType.STANDARD,
        risk_config: RiskConfig | None = None
    ):
        """
        Args:
            node_id: 노드 식별자 (None이면 자동 생성)
            version: 토큰 버전
            token_type: 토큰 타입
            risk_config: 위험 점수 설정
        """
        self.node_id = node_id or uuid.uuid4().bytes
        self.version = version
        self.token_type = token_type
        self.risk_scorer = RiskScorer(risk_config)
        
        # 파이프라인 구성요소
        self.feature_extractor = FeatureExtractor()
        self.correlation_sig = CorrelationSignature()
    
    def from_sensor_data(
        self, 
        sensor_data: dict[str, np.ndarray],
        timestamp_ns: int | None = None
    ) -> PoXToken:
        """
        센서 데이터에서 토큰 생성
        
        Args:
            sensor_data: {sensor_name: time_series} 딕셔너리
            timestamp_ns: 타임스탬프 (None이면 현재 시간)
        
        Returns:
            생성된 PoXToken
        """
        # 1. 특징 추출
        features = self.feature_extractor.extract_multi(sensor_data)
        
        # 2. 상관 서명 계산
        correlation = self.correlation_sig.compute(sensor_data)
        
        # 3. 위험 점수 계산
        risk_score = self.risk_scorer.compute(features, correlation)
        
        # 4. 지문 생성
        fingerprint = self._create_fingerprint(features, correlation)
        
        # 5. 토큰 조립
        token_kwargs = {
            "node_id": self.node_id,
            "fingerprint": fingerprint,
            "risk_score": risk_score,
            "version": self.version,
            "token_type": self.token_type,
        }
        
        if timestamp_ns is not None:
            token_kwargs["timestamp_ns"] = timestamp_ns
        
        return PoXToken(**token_kwargs)
    
    def from_correlation_result(
        self,
        correlation: CorrelationResult,
        feature_vector_bytes: bytes | None = None,
        risk_score: float = 0.0,
        timestamp_ns: int | None = None
    ) -> PoXToken:
        """
        상관 결과에서 직접 토큰 생성 (고급 사용)
        
        Args:
            correlation: 상관 서명 결과
            feature_vector_bytes: 특징 벡터 바이트 (None이면 correlation에서)
            risk_score: 위험 점수
            timestamp_ns: 타임스탬프
        
        Returns:
            생성된 PoXToken
        """
        # 특징 벡터 변환
        if feature_vector_bytes is None:
            fv = correlation.feature_vector
            # float32 배열로 패딩 (16개)
            padded = np.zeros(16, dtype=np.float32)
            padded[:min(len(fv), 16)] = fv[:16]
            feature_vector_bytes = padded.tobytes()
        
        fingerprint = NoiseFingerprint(
            feature_vector=feature_vector_bytes,
            correlation_hash=correlation.correlation_hash,
            sensor_count=correlation.sensor_count,
            sample_count=256  # 기본값
        )
        
        token_kwargs = {
            "node_id": self.node_id,
            "fingerprint": fingerprint,
            "risk_score": risk_score,
            "version": self.version,
            "token_type": self.token_type,
        }
        
        if timestamp_ns is not None:
            token_kwargs["timestamp_ns"] = timestamp_ns
        
        return PoXToken(**token_kwargs)
    
    def _create_fingerprint(
        self, 
        features: dict[str, FeatureVector],
        correlation: CorrelationResult
    ) -> NoiseFingerprint:
        """NoiseFingerprint 생성"""
        # 특징 벡터 결합 (각 센서 주요 통계)
        from noisechain.pipeline.features import FeatureType
        
        feature_list = []
        for name, fv in features.items():
            feature_list.extend([
                fv.get(FeatureType.MEAN, 0),
                fv.get(FeatureType.VARIANCE, 0),
                fv.get(FeatureType.ENTROPY, 0),
                fv.get(FeatureType.RMS, 0),
            ])
        
        # 16개로 맞추기
        padded = np.zeros(16, dtype=np.float32)
        padded[:min(len(feature_list), 16)] = feature_list[:16]
        
        return NoiseFingerprint(
            feature_vector=padded.tobytes(),
            correlation_hash=correlation.correlation_hash,
            sensor_count=correlation.sensor_count,
            sample_count=min(65535, max(1, sum(
                fv.source_length for fv in features.values()
            )))
        )


def build_token(
    sensor_data: dict[str, np.ndarray],
    node_id: bytes | None = None
) -> PoXToken:
    """
    편의 함수: 센서 데이터에서 토큰 생성
    
    Args:
        sensor_data: 센서 데이터
        node_id: 노드 ID (None이면 자동 생성)
    
    Returns:
        PoXToken
    """
    builder = TokenBuilder(node_id=node_id)
    return builder.from_sensor_data(sensor_data)
