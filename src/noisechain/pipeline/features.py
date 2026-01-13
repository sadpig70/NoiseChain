"""
NoiseChain 특징 추출기

시계열 노이즈 데이터에서 통계적, 주파수, 시간 특징을 추출합니다.
이 특징들은 상관 서명 계산과 노이즈 지문 생성에 사용됩니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy import stats, signal


class FeatureType(Enum):
    """특징 타입 열거형"""
    # 통계적 특징
    MEAN = auto()
    VARIANCE = auto()
    STD = auto()
    SKEWNESS = auto()
    KURTOSIS = auto()
    ENTROPY = auto()
    MIN = auto()
    MAX = auto()
    RANGE = auto()
    MEDIAN = auto()
    
    # 주파수 특징
    DOMINANT_FREQ = auto()
    SPECTRAL_CENTROID = auto()
    SPECTRAL_BANDWIDTH = auto()
    TOTAL_POWER = auto()
    BAND_POWER_LOW = auto()
    BAND_POWER_MID = auto()
    BAND_POWER_HIGH = auto()
    
    # 시간 특징
    ZERO_CROSSING_RATE = auto()
    AUTOCORR_LAG1 = auto()
    AUTOCORR_LAG5 = auto()
    RMS = auto()
    PEAK_TO_PEAK = auto()


@dataclass
class FeatureConfig:
    """특징 추출 설정"""
    # 통계 특징
    compute_stats: bool = True
    entropy_bins: int = 64          # 엔트로피 계산용 빈 수
    
    # 주파수 특징
    compute_freq: bool = True
    sampling_rate_hz: int = 100     # 샘플링 레이트
    fft_size: Optional[int] = None  # FFT 크기 (None이면 자동)
    
    # 시간 특징
    compute_temporal: bool = True
    autocorr_lags: list[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class FeatureVector:
    """
    추출된 특징 벡터
    
    Attributes:
        values: 특징 타입별 값 딕셔너리
        source_length: 원본 시계열 길이
        sampling_rate: 샘플링 레이트
    """
    values: dict[FeatureType, float] = field(default_factory=dict)
    source_length: int = 0
    sampling_rate: int = 100
    
    def get(self, feature_type: FeatureType, default: float = 0.0) -> float:
        """특정 특징 값 조회"""
        return self.values.get(feature_type, default)
    
    def to_array(self, feature_order: Optional[list[FeatureType]] = None) -> np.ndarray:
        """
        numpy 배열로 변환
        
        Args:
            feature_order: 특징 순서 (None이면 FeatureType 정의 순서)
        
        Returns:
            특징 값 배열
        """
        if feature_order is None:
            feature_order = list(FeatureType)
        
        return np.array([self.values.get(ft, 0.0) for ft in feature_order])
    
    def to_normalized_array(
        self, 
        feature_order: Optional[list[FeatureType]] = None,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        정규화된 numpy 배열로 변환
        
        Args:
            feature_order: 특징 순서
            method: 정규화 방법 ("minmax" 또는 "zscore")
        
        Returns:
            정규화된 특징 값 배열
        """
        arr = self.to_array(feature_order)
        
        if method == "minmax":
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min > 1e-10:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.zeros_like(arr)
        elif method == "zscore":
            std = arr.std()
            if std > 1e-10:
                return (arr - arr.mean()) / std
            return np.zeros_like(arr)
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")
    
    @property
    def size(self) -> int:
        """특징 개수"""
        return len(self.values)
    
    def __repr__(self) -> str:
        return f"FeatureVector(size={self.size}, source_length={self.source_length})"


class FeatureExtractor:
    """
    특징 추출기
    
    시계열 데이터에서 다양한 특징을 추출합니다.
    
    Attributes:
        config: 추출 설정
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(signal_data)
        >>> mean_val = features.get(FeatureType.MEAN)
    """
    
    def __init__(self, config: FeatureConfig | None = None):
        """
        Args:
            config: 추출 설정 (None이면 기본값 사용)
        """
        self.config = config or FeatureConfig()
    
    def extract(self, data: np.ndarray) -> FeatureVector:
        """
        특징 추출
        
        Args:
            data: 1D 시계열 데이터
        
        Returns:
            추출된 특징 벡터
        """
        if len(data) == 0:
            return FeatureVector()
        
        # 데이터 전처리
        data = np.asarray(data, dtype=np.float64)
        
        result = FeatureVector(
            source_length=len(data),
            sampling_rate=self.config.sampling_rate_hz
        )
        
        # 통계 특징
        if self.config.compute_stats:
            self._extract_statistical(data, result)
        
        # 주파수 특징
        if self.config.compute_freq:
            self._extract_frequency(data, result)
        
        # 시간 특징
        if self.config.compute_temporal:
            self._extract_temporal(data, result)
        
        return result
    
    def extract_multi(self, data_dict: dict[str, np.ndarray]) -> dict[str, FeatureVector]:
        """
        다중 시계열 특징 추출
        
        Args:
            data_dict: 이름별 시계열 딕셔너리
        
        Returns:
            이름별 특징 벡터 딕셔너리
        """
        return {name: self.extract(data) for name, data in data_dict.items()}
    
    def _extract_statistical(self, data: np.ndarray, result: FeatureVector) -> None:
        """통계적 특징 추출"""
        # 기본 통계
        result.values[FeatureType.MEAN] = float(np.mean(data))
        result.values[FeatureType.VARIANCE] = float(np.var(data))
        result.values[FeatureType.STD] = float(np.std(data))
        result.values[FeatureType.MIN] = float(np.min(data))
        result.values[FeatureType.MAX] = float(np.max(data))
        result.values[FeatureType.RANGE] = float(np.ptp(data))  # peak-to-peak
        result.values[FeatureType.MEDIAN] = float(np.median(data))
        
        # 고차 통계량
        if len(data) > 2:
            result.values[FeatureType.SKEWNESS] = float(stats.skew(data))
            result.values[FeatureType.KURTOSIS] = float(stats.kurtosis(data))
        else:
            result.values[FeatureType.SKEWNESS] = 0.0
            result.values[FeatureType.KURTOSIS] = 0.0
        
        # 엔트로피 (히스토그램 기반)
        result.values[FeatureType.ENTROPY] = self._compute_entropy(data)
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """샤논 엔트로피 계산"""
        # 히스토그램 생성
        hist, _ = np.histogram(data, bins=self.config.entropy_bins, density=True)
        
        # 0 제거 및 정규화
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        
        # 엔트로피 계산: -sum(p * log2(p))
        bin_width = (data.max() - data.min()) / self.config.entropy_bins
        if bin_width > 0:
            hist = hist * bin_width  # 확률로 변환
            return float(-np.sum(hist * np.log2(hist + 1e-10)))
        return 0.0
    
    def _extract_frequency(self, data: np.ndarray, result: FeatureVector) -> None:
        """주파수 특징 추출"""
        n = len(data)
        if n < 4:
            # 데이터 부족 시 기본값
            for ft in [FeatureType.DOMINANT_FREQ, FeatureType.SPECTRAL_CENTROID,
                      FeatureType.SPECTRAL_BANDWIDTH, FeatureType.TOTAL_POWER,
                      FeatureType.BAND_POWER_LOW, FeatureType.BAND_POWER_MID,
                      FeatureType.BAND_POWER_HIGH]:
                result.values[ft] = 0.0
            return
        
        # FFT 계산
        fft_size = self.config.fft_size or n
        fft_result = np.fft.rfft(data, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, d=1.0/self.config.sampling_rate_hz)
        magnitudes = np.abs(fft_result)
        power = magnitudes ** 2
        
        # 총 파워
        total_power = np.sum(power)
        result.values[FeatureType.TOTAL_POWER] = float(total_power)
        
        if total_power > 1e-10:
            # 지배 주파수 (최대 파워 주파수)
            dominant_idx = np.argmax(magnitudes[1:]) + 1  # DC 제외
            result.values[FeatureType.DOMINANT_FREQ] = float(freqs[dominant_idx])
            
            # 스펙트럼 중심 (가중 평균 주파수)
            spectral_centroid = np.sum(freqs * power) / total_power
            result.values[FeatureType.SPECTRAL_CENTROID] = float(spectral_centroid)
            
            # 스펙트럼 대역폭 (표준편차)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)
            result.values[FeatureType.SPECTRAL_BANDWIDTH] = float(spectral_bandwidth)
        else:
            result.values[FeatureType.DOMINANT_FREQ] = 0.0
            result.values[FeatureType.SPECTRAL_CENTROID] = 0.0
            result.values[FeatureType.SPECTRAL_BANDWIDTH] = 0.0
        
        # 대역 파워 (저/중/고)
        nyquist = self.config.sampling_rate_hz / 2
        low_mask = freqs < nyquist * 0.2
        mid_mask = (freqs >= nyquist * 0.2) & (freqs < nyquist * 0.6)
        high_mask = freqs >= nyquist * 0.6
        
        result.values[FeatureType.BAND_POWER_LOW] = float(np.sum(power[low_mask]))
        result.values[FeatureType.BAND_POWER_MID] = float(np.sum(power[mid_mask]))
        result.values[FeatureType.BAND_POWER_HIGH] = float(np.sum(power[high_mask]))
    
    def _extract_temporal(self, data: np.ndarray, result: FeatureVector) -> None:
        """시간 특징 추출"""
        n = len(data)
        
        # 영교차율 (Zero-Crossing Rate)
        if n > 1:
            zero_crossings = np.sum(np.diff(np.signbit(data)))
            result.values[FeatureType.ZERO_CROSSING_RATE] = float(zero_crossings / (n - 1))
        else:
            result.values[FeatureType.ZERO_CROSSING_RATE] = 0.0
        
        # RMS (Root Mean Square)
        result.values[FeatureType.RMS] = float(np.sqrt(np.mean(data ** 2)))
        
        # Peak-to-Peak
        result.values[FeatureType.PEAK_TO_PEAK] = float(np.ptp(data))
        
        # 자기상관 (Lag 1, 5)
        if n > 5:
            autocorr = self._compute_autocorrelation(data, max_lag=10)
            result.values[FeatureType.AUTOCORR_LAG1] = float(autocorr[1]) if len(autocorr) > 1 else 0.0
            result.values[FeatureType.AUTOCORR_LAG5] = float(autocorr[5]) if len(autocorr) > 5 else 0.0
        else:
            result.values[FeatureType.AUTOCORR_LAG1] = 0.0
            result.values[FeatureType.AUTOCORR_LAG5] = 0.0
    
    def _compute_autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """자기상관 계산"""
        n = len(data)
        if n < 2:
            return np.zeros(max_lag + 1)
        
        # 정규화된 자기상관
        mean = np.mean(data)
        var = np.var(data)
        if var < 1e-10:
            return np.zeros(max_lag + 1)
        
        autocorr = np.zeros(min(max_lag + 1, n))
        data_centered = data - mean
        
        for lag in range(len(autocorr)):
            if lag == 0:
                autocorr[0] = 1.0
            else:
                autocorr[lag] = np.sum(data_centered[:-lag] * data_centered[lag:]) / (var * (n - lag))
        
        return autocorr


def extract_features(data: np.ndarray, config: FeatureConfig | None = None) -> FeatureVector:
    """
    편의 함수: 특징 추출
    
    Args:
        data: 1D 시계열 데이터
        config: 추출 설정
    
    Returns:
        특징 벡터
    """
    extractor = FeatureExtractor(config)
    return extractor.extract(data)
