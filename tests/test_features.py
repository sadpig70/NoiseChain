"""
NoiseChain 특징 추출기 테스트

FeatureExtractor 및 관련 클래스의 단위 테스트입니다.
"""

import numpy as np
import pytest

from noisechain.pipeline.features import (
    FeatureConfig,
    FeatureExtractor,
    FeatureType,
    FeatureVector,
    extract_features,
)


# ============================================================
# FeatureType 테스트
# ============================================================

class TestFeatureType:
    """FeatureType 열거형 테스트"""
    
    def test_statistical_features_exist(self):
        """통계 특징 정의 확인"""
        assert FeatureType.MEAN is not None
        assert FeatureType.VARIANCE is not None
        assert FeatureType.SKEWNESS is not None
        assert FeatureType.ENTROPY is not None
    
    def test_frequency_features_exist(self):
        """주파수 특징 정의 확인"""
        assert FeatureType.DOMINANT_FREQ is not None
        assert FeatureType.SPECTRAL_CENTROID is not None
        assert FeatureType.TOTAL_POWER is not None
    
    def test_temporal_features_exist(self):
        """시간 특징 정의 확인"""
        assert FeatureType.ZERO_CROSSING_RATE is not None
        assert FeatureType.AUTOCORR_LAG1 is not None
        assert FeatureType.RMS is not None


# ============================================================
# FeatureVector 테스트
# ============================================================

class TestFeatureVector:
    """FeatureVector 테스트"""
    
    def test_empty_initialization(self):
        """빈 벡터 초기화"""
        fv = FeatureVector()
        
        assert fv.size == 0
        assert fv.source_length == 0
    
    def test_get_existing_value(self):
        """존재하는 특징 값 조회"""
        fv = FeatureVector(values={
            FeatureType.MEAN: 10.0,
            FeatureType.VARIANCE: 5.0
        })
        
        assert fv.get(FeatureType.MEAN) == 10.0
        assert fv.get(FeatureType.VARIANCE) == 5.0
    
    def test_get_missing_value_default(self):
        """없는 특징 값 기본값"""
        fv = FeatureVector()
        
        assert fv.get(FeatureType.MEAN) == 0.0
        assert fv.get(FeatureType.MEAN, default=999.0) == 999.0
    
    def test_to_array(self):
        """numpy 배열 변환"""
        fv = FeatureVector(values={
            FeatureType.MEAN: 1.0,
            FeatureType.VARIANCE: 2.0
        })
        
        arr = fv.to_array(feature_order=[FeatureType.MEAN, FeatureType.VARIANCE])
        
        np.testing.assert_array_equal(arr, [1.0, 2.0])
    
    def test_to_normalized_array_minmax(self):
        """MinMax 정규화"""
        fv = FeatureVector(values={
            FeatureType.MEAN: 0.0,
            FeatureType.VARIANCE: 10.0
        })
        
        arr = fv.to_normalized_array(
            feature_order=[FeatureType.MEAN, FeatureType.VARIANCE],
            method="minmax"
        )
        
        np.testing.assert_array_almost_equal(arr, [0.0, 1.0])
    
    def test_to_normalized_array_zscore(self):
        """Z-score 정규화"""
        fv = FeatureVector(values={
            FeatureType.MEAN: 0.0,
            FeatureType.VARIANCE: 2.0,
            FeatureType.STD: 4.0
        })
        
        arr = fv.to_normalized_array(
            feature_order=[FeatureType.MEAN, FeatureType.VARIANCE, FeatureType.STD],
            method="zscore"
        )
        
        assert abs(arr.mean()) < 1e-10  # 평균 0
        assert abs(arr.std() - 1.0) < 1e-10  # 표준편차 1
    
    def test_size_property(self):
        """크기 속성"""
        fv = FeatureVector(values={
            FeatureType.MEAN: 1.0,
            FeatureType.VARIANCE: 2.0,
            FeatureType.STD: 3.0
        })
        
        assert fv.size == 3


# ============================================================
# FeatureConfig 테스트
# ============================================================

class TestFeatureConfig:
    """FeatureConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        config = FeatureConfig()
        
        assert config.compute_stats is True
        assert config.compute_freq is True
        assert config.compute_temporal is True
        assert config.sampling_rate_hz == 100
    
    def test_custom_values(self):
        """사용자 정의 값"""
        config = FeatureConfig(
            sampling_rate_hz=200,
            entropy_bins=32
        )
        
        assert config.sampling_rate_hz == 200
        assert config.entropy_bins == 32


# ============================================================
# FeatureExtractor 테스트
# ============================================================

class TestFeatureExtractor:
    """FeatureExtractor 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        extractor = FeatureExtractor()
        
        assert extractor.config is not None
    
    def test_extract_empty_data(self):
        """빈 데이터 처리"""
        extractor = FeatureExtractor()
        
        result = extractor.extract(np.array([]))
        
        assert result.size == 0
    
    def test_extract_single_value(self):
        """단일 값 처리"""
        extractor = FeatureExtractor()
        
        result = extractor.extract(np.array([5.0]))
        
        assert result.get(FeatureType.MEAN) == 5.0
        assert result.get(FeatureType.VARIANCE) == 0.0
    
    def test_extract_statistical_features(self):
        """통계 특징 추출"""
        extractor = FeatureExtractor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = extractor.extract(data)
        
        assert result.get(FeatureType.MEAN) == 3.0
        assert result.get(FeatureType.MIN) == 1.0
        assert result.get(FeatureType.MAX) == 5.0
        assert result.get(FeatureType.RANGE) == 4.0
        assert result.get(FeatureType.MEDIAN) == 3.0
    
    def test_extract_variance_std(self):
        """분산/표준편차 확인"""
        extractor = FeatureExtractor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = extractor.extract(data)
        
        expected_var = np.var(data)
        expected_std = np.std(data)
        
        assert abs(result.get(FeatureType.VARIANCE) - expected_var) < 1e-10
        assert abs(result.get(FeatureType.STD) - expected_std) < 1e-10
    
    def test_extract_skewness_kurtosis(self):
        """왜도/첨도 추출"""
        extractor = FeatureExtractor()
        # 정규분포 유사 데이터
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        result = extractor.extract(data)
        
        # 정규분포의 왜도 ≈ 0, 첨도 ≈ 0 (excess kurtosis)
        assert abs(result.get(FeatureType.SKEWNESS)) < 0.5
        assert abs(result.get(FeatureType.KURTOSIS)) < 0.5
    
    def test_extract_entropy(self):
        """엔트로피 추출"""
        extractor = FeatureExtractor()
        # 균일 분포 데이터 (높은 엔트로피)
        uniform_data = np.linspace(0, 1, 100)
        
        result = extractor.extract(uniform_data)
        
        assert result.get(FeatureType.ENTROPY) > 0
    
    def test_extract_frequency_features(self):
        """주파수 특징 추출"""
        config = FeatureConfig(sampling_rate_hz=100)
        extractor = FeatureExtractor(config)
        
        # 10Hz 사인파
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 10 * t)
        
        result = extractor.extract(data)
        
        # 지배 주파수 ≈ 10Hz
        dominant_freq = result.get(FeatureType.DOMINANT_FREQ)
        assert 8.0 <= dominant_freq <= 12.0
        
        # 총 파워 > 0
        assert result.get(FeatureType.TOTAL_POWER) > 0
    
    def test_extract_band_power(self):
        """대역 파워 추출"""
        config = FeatureConfig(sampling_rate_hz=100)
        extractor = FeatureExtractor(config)
        
        # 저역 신호 (5Hz)
        t = np.linspace(0, 1, 100)
        low_freq_signal = np.sin(2 * np.pi * 5 * t)
        
        result = extractor.extract(low_freq_signal)
        
        # 저역 파워가 가장 높아야 함
        low = result.get(FeatureType.BAND_POWER_LOW)
        mid = result.get(FeatureType.BAND_POWER_MID)
        high = result.get(FeatureType.BAND_POWER_HIGH)
        
        assert low > mid
        assert low > high
    
    def test_extract_temporal_features(self):
        """시간 특징 추출"""
        extractor = FeatureExtractor()
        data = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        
        result = extractor.extract(data)
        
        # 영교차율 = 1.0 (매번 교차)
        assert result.get(FeatureType.ZERO_CROSSING_RATE) == 1.0
    
    def test_extract_rms(self):
        """RMS 추출"""
        extractor = FeatureExtractor()
        data = np.array([3.0, 4.0])  # RMS = sqrt((9+16)/2) = sqrt(12.5)
        
        result = extractor.extract(data)
        
        expected_rms = np.sqrt(12.5)
        assert abs(result.get(FeatureType.RMS) - expected_rms) < 1e-10
    
    def test_extract_autocorrelation(self):
        """자기상관 추출"""
        extractor = FeatureExtractor()
        # 주기적 신호
        data = np.array([1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0])
        
        result = extractor.extract(data)
        
        # lag-1 자기상관 존재
        assert FeatureType.AUTOCORR_LAG1 in result.values
    
    def test_extract_multi(self):
        """다중 시계열 추출"""
        extractor = FeatureExtractor()
        data_dict = {
            "sensor1": np.array([1.0, 2.0, 3.0]),
            "sensor2": np.array([4.0, 5.0, 6.0])
        }
        
        results = extractor.extract_multi(data_dict)
        
        assert "sensor1" in results
        assert "sensor2" in results
        assert results["sensor1"].get(FeatureType.MEAN) == 2.0
        assert results["sensor2"].get(FeatureType.MEAN) == 5.0
    
    def test_disable_feature_groups(self):
        """특징 그룹 비활성화"""
        config = FeatureConfig(
            compute_stats=True,
            compute_freq=False,
            compute_temporal=False
        )
        extractor = FeatureExtractor(config)
        
        result = extractor.extract(np.linspace(0, 1, 100))
        
        # 통계 특징만 존재
        assert FeatureType.MEAN in result.values
        assert FeatureType.DOMINANT_FREQ not in result.values
        assert FeatureType.ZERO_CROSSING_RATE not in result.values


class TestExtractFeaturesFunction:
    """extract_features 편의 함수 테스트"""
    
    def test_basic_usage(self):
        """기본 사용법"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = extract_features(data)
        
        assert result.get(FeatureType.MEAN) == 3.0
    
    def test_with_config(self):
        """설정과 함께 사용"""
        config = FeatureConfig(compute_freq=False)
        data = np.array([1.0, 2.0, 3.0])
        
        result = extract_features(data, config)
        
        assert FeatureType.DOMINANT_FREQ not in result.values


class TestIntegrationFeatures:
    """특징 추출 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우"""
        # 복합 신호 생성
        np.random.seed(42)
        t = np.linspace(0, 1, 256)
        signal = (
            np.sin(2 * np.pi * 10 * t) +    # 10Hz 성분
            0.5 * np.sin(2 * np.pi * 25 * t) +  # 25Hz 성분
            0.1 * np.random.randn(256)      # 노이즈
        )
        
        # 추출
        config = FeatureConfig(sampling_rate_hz=256)
        extractor = FeatureExtractor(config)
        result = extractor.extract(signal)
        
        # 모든 특징 타입 존재 확인
        expected_features = [
            FeatureType.MEAN, FeatureType.VARIANCE, FeatureType.ENTROPY,
            FeatureType.DOMINANT_FREQ, FeatureType.SPECTRAL_CENTROID,
            FeatureType.ZERO_CROSSING_RATE, FeatureType.RMS
        ]
        
        for ft in expected_features:
            assert ft in result.values, f"{ft.name} 누락"
        
        # 배열 변환
        arr = result.to_array(expected_features)
        assert arr.shape == (len(expected_features),)
        
        # 정규화
        normalized = result.to_normalized_array(expected_features, method="minmax")
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
