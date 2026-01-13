"""
NoiseChain 상관 서명 알고리즘

다중 센서 시계열 데이터 간의 교차 상관을 계산하고,
이를 양자화하여 고유한 상관 해시(Correlation Hash)를 생성합니다.

설계 기반: NoiseChain_MVP_Design.md 섹션 5 (CorrelationSignature)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class CorrelationConfig:
    """상관 서명 설정"""
    max_lag: int = 10              # 최대 라그 (샘플 수)
    lag_step: int = 1              # 라그 간격
    quant_bits: int = 8            # 양자화 비트 수
    feature_size: int = 16         # 최종 특징 벡터 크기
    similarity_threshold: float = 0.85  # 검증 시 최소 유사도


@dataclass
class CorrelationResult:
    """
    상관 서명 결과
    
    Attributes:
        feature_vector: 정규화된 특징 벡터 [0, 1]
        correlation_hash: SHA3-256 해시 (32 bytes)
        sensor_count: 사용된 센서 개수
        correlation_matrix: 전체 상관 행렬 (선택적)
    """
    feature_vector: np.ndarray    # shape: (feature_size,)
    correlation_hash: bytes       # 32 bytes
    sensor_count: int
    correlation_matrix: Optional[np.ndarray] = None  # shape: (n, n, lags)
    
    def to_dict(self) -> dict:
        """딕셔너리 변환 (직렬화용)"""
        return {
            "feature_vector": self.feature_vector.tolist(),
            "correlation_hash": self.correlation_hash.hex(),
            "sensor_count": self.sensor_count,
        }


class CorrelationSignature:
    """
    상관 서명 알고리즘
    
    다중 센서 시계열 간 교차 상관을 계산하고v
    고유한 노이즈 지문(Noise Fingerprint)을 생성합니다.
    
    알고리즘 흐름:
    1. 다중 센서 시계열 입력
    2. 모든 센서 쌍에 대해 교차 상관 계산
    3. 상관 행렬 정규화 (MinMax)
    4. 특징 벡터 추출 (상삼각 + 대각선)
    5. 양자화 (8비트)
    6. SHA3-256 해시 생성
    
    Example:
        >>> sig = CorrelationSignature()
        >>> result = sig.compute({"sensor1": data1, "sensor2": data2})
        >>> is_valid, sim = sig.verify(result, new_result)
    """
    
    def __init__(self, config: CorrelationConfig | None = None):
        """
        Args:
            config: 상관 서명 설정 (None이면 기본값 사용)
        """
        self.config = config or CorrelationConfig()
    
    def compute(
        self, 
        sensor_data: dict[str, np.ndarray],
        return_matrix: bool = False
    ) -> CorrelationResult:
        """
        상관 서명 계산
        
        Args:
            sensor_data: {sensor_name: time_series} 딕셔너리
            return_matrix: 전체 상관 행렬 포함 여부
        
        Returns:
            CorrelationResult 객체
        """
        sensors = list(sensor_data.keys())
        n_sensors = len(sensors)
        
        if n_sensors == 0:
            return CorrelationResult(
                feature_vector=np.zeros(self.config.feature_size),
                correlation_hash=hashlib.sha3_256(b"empty").digest(),
                sensor_count=0
            )
        
        if n_sensors == 1:
            # 단일 센서: 자기상관만
            single_data = sensor_data[sensors[0]]
            autocorr = self._compute_autocorrelation(single_data)
            feature_vector = self._extract_single_features(autocorr)
            
            return CorrelationResult(
                feature_vector=feature_vector,
                correlation_hash=self._compute_hash(feature_vector, autocorr),
                sensor_count=1
            )
        
        # 라그 범위
        lags = list(range(-self.config.max_lag, self.config.max_lag + 1, self.config.lag_step))
        n_lags = len(lags)
        
        # 1. 교차 상관 행렬 계산 (N x N x L)
        corr_matrix = np.zeros((n_sensors, n_sensors, n_lags))
        
        for i, s1 in enumerate(sensors):
            for j, s2 in enumerate(sensors):
                if i <= j:
                    corr = self._compute_cross_correlation(
                        sensor_data[s1], 
                        sensor_data[s2]
                    )
                    corr_matrix[i, j, :] = corr
                    if i != j:
                        corr_matrix[j, i, :] = corr  # 대칭
        
        # 2. 정규화 (MinMax)
        normalized = self._normalize_matrix(corr_matrix)
        
        # 3. 특징 벡터 추출
        feature_vector = self._extract_features(normalized, n_sensors)
        
        # 4. 해시 생성
        correlation_hash = self._compute_hash(feature_vector, normalized)
        
        return CorrelationResult(
            feature_vector=feature_vector,
            correlation_hash=correlation_hash,
            sensor_count=n_sensors,
            correlation_matrix=corr_matrix if return_matrix else None
        )
    
    def verify(
        self,
        original: CorrelationResult,
        new: CorrelationResult
    ) -> tuple[bool, float]:
        """
        상관 서명 검증
        
        원본과 새 서명의 유사도를 계산하여 일치 여부 판정.
        
        Args:
            original: 저장된 원본 결과
            new: 검증할 새 결과
        
        Returns:
            (is_valid, similarity_score)
        """
        # 해시 일치 확인
        hash_match = (original.correlation_hash == new.correlation_hash)
        
        # 특징 벡터 유사도 (코사인 유사도)
        similarity = self._cosine_similarity(
            original.feature_vector, 
            new.feature_vector
        )
        
        # 유효성 판정: 해시 일치 또는 유사도 임계값 이상
        is_valid = hash_match or (similarity >= self.config.similarity_threshold)
        
        return is_valid, float(similarity)
    
    def _compute_cross_correlation(
        self, 
        signal1: np.ndarray, 
        signal2: np.ndarray
    ) -> np.ndarray:
        """두 신호 간 교차 상관 계산"""
        # 데이터 정규화
        s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
        
        # scipy.signal.correlate 사용
        corr = signal.correlate(s1, s2, mode='same')
        
        # 중앙 ±max_lag 추출
        center = len(corr) // 2
        max_lag = self.config.max_lag
        
        start = max(0, center - max_lag)
        end = min(len(corr), center + max_lag + 1)
        
        result = corr[start:end]
        
        # 길이 맞추기
        expected_len = 2 * max_lag + 1
        if len(result) < expected_len:
            padded = np.zeros(expected_len)
            offset = (expected_len - len(result)) // 2
            padded[offset:offset + len(result)] = result
            result = padded
        elif len(result) > expected_len:
            result = result[:expected_len]
        
        return result / (len(signal1) + 1e-10)  # 정규화
    
    def _compute_autocorrelation(self, data: np.ndarray) -> np.ndarray:
        """자기상관 계산"""
        return self._compute_cross_correlation(data, data)
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """MinMax 정규화"""
        mat_min = matrix.min()
        mat_max = matrix.max()
        
        if mat_max - mat_min < 1e-10:
            return np.zeros_like(matrix)
        
        return (matrix - mat_min) / (mat_max - mat_min)
    
    def _extract_features(
        self, 
        normalized: np.ndarray, 
        n_sensors: int
    ) -> np.ndarray:
        """특징 벡터 추출 (상삼각 + 대각선의 최대 상관값)"""
        feature_list = []
        
        for i in range(n_sensors):
            for j in range(i, n_sensors):
                # 각 센서 쌍의 최대 상관 라그와 값
                max_idx = np.argmax(np.abs(normalized[i, j, :]))
                feature_list.append(normalized[i, j, max_idx])
        
        # feature_size로 자르거나 패딩
        features = np.array(feature_list[:self.config.feature_size])
        
        if len(features) < self.config.feature_size:
            padded = np.zeros(self.config.feature_size)
            padded[:len(features)] = features
            features = padded
        
        return features
    
    def _extract_single_features(self, autocorr: np.ndarray) -> np.ndarray:
        """단일 센서 특징 추출"""
        features = np.zeros(self.config.feature_size)
        
        # 자기상관의 주요 특징
        if len(autocorr) > 0:
            features[0] = autocorr.max()
            features[1] = autocorr.min()
            features[2] = autocorr.mean()
            features[3] = autocorr.std()
            
            # 피크 위치
            peak_idx = np.argmax(np.abs(autocorr))
            features[4] = peak_idx / len(autocorr) if len(autocorr) > 0 else 0
        
        return features
    
    def _compute_hash(
        self, 
        feature_vector: np.ndarray, 
        normalized_data: np.ndarray
    ) -> bytes:
        """SHA3-256 해시 생성"""
        # 양자화 (8비트)
        quantized = np.round(
            feature_vector * (2**self.config.quant_bits - 1)
        ).astype(np.uint8)
        
        # 해시 입력: 양자화된 특징 + 정규화 데이터
        hash_input = quantized.tobytes() + normalized_data.tobytes()
        
        return hashlib.sha3_256(hash_input).digest()
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return dot / (norm1 * norm2)


def compute_correlation_signature(
    sensor_data: dict[str, np.ndarray],
    config: CorrelationConfig | None = None
) -> CorrelationResult:
    """
    편의 함수: 상관 서명 계산
    
    Args:
        sensor_data: 센서별 시계열 데이터
        config: 설정
    
    Returns:
        CorrelationResult
    """
    sig = CorrelationSignature(config)
    return sig.compute(sensor_data)
