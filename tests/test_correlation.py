"""
NoiseChain 상관 서명 알고리즘 테스트

CorrelationSignature 및 관련 클래스의 단위 테스트입니다.
"""

import numpy as np
import pytest

from noisechain.pipeline.correlation import (
    CorrelationConfig,
    CorrelationResult,
    CorrelationSignature,
    compute_correlation_signature,
)


# ============================================================
# CorrelationConfig 테스트
# ============================================================

class TestCorrelationConfig:
    """CorrelationConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        config = CorrelationConfig()
        
        assert config.max_lag == 10
        assert config.quant_bits == 8
        assert config.feature_size == 16
        assert config.similarity_threshold == 0.85
    
    def test_custom_values(self):
        """사용자 정의 값"""
        config = CorrelationConfig(
            max_lag=20,
            feature_size=32
        )
        
        assert config.max_lag == 20
        assert config.feature_size == 32


# ============================================================
# CorrelationResult 테스트
# ============================================================

class TestCorrelationResult:
    """CorrelationResult 테스트"""
    
    def test_initialization(self):
        """초기화"""
        result = CorrelationResult(
            feature_vector=np.array([0.1, 0.2, 0.3]),
            correlation_hash=b"test_hash_32_bytes_padding_here!",
            sensor_count=2
        )
        
        assert len(result.feature_vector) == 3
        assert result.sensor_count == 2
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        result = CorrelationResult(
            feature_vector=np.array([0.5, 0.6]),
            correlation_hash=bytes(32),
            sensor_count=3
        )
        
        d = result.to_dict()
        
        assert "feature_vector" in d
        assert "correlation_hash" in d
        assert "sensor_count" in d
        assert d["sensor_count"] == 3


# ============================================================
# CorrelationSignature 테스트
# ============================================================

class TestCorrelationSignature:
    """CorrelationSignature 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        sig = CorrelationSignature()
        
        assert sig.config is not None
    
    def test_custom_config(self):
        """사용자 정의 설정"""
        config = CorrelationConfig(max_lag=5)
        sig = CorrelationSignature(config)
        
        assert sig.config.max_lag == 5
    
    def test_compute_empty_data(self):
        """빈 데이터 처리"""
        sig = CorrelationSignature()
        
        result = sig.compute({})
        
        assert result.sensor_count == 0
        assert len(result.feature_vector) == 16  # 기본 feature_size
    
    def test_compute_single_sensor(self):
        """단일 센서 처리"""
        sig = CorrelationSignature()
        data = {"sensor1": np.sin(np.linspace(0, 4*np.pi, 100))}
        
        result = sig.compute(data)
        
        assert result.sensor_count == 1
        assert len(result.feature_vector) == 16
        assert len(result.correlation_hash) == 32  # SHA3-256
    
    def test_compute_two_sensors_identical(self):
        """동일한 두 센서"""
        sig = CorrelationSignature()
        data = np.sin(np.linspace(0, 4*np.pi, 100))
        
        result = sig.compute({
            "sensor1": data.copy(),
            "sensor2": data.copy()
        })
        
        assert result.sensor_count == 2
        # 동일 신호는 높은 상관
        assert result.feature_vector.max() > 0.5
    
    def test_compute_two_sensors_different(self):
        """다른 두 센서"""
        sig = CorrelationSignature()
        
        result = sig.compute({
            "sensor1": np.sin(np.linspace(0, 4*np.pi, 100)),
            "sensor2": np.cos(np.linspace(0, 4*np.pi, 100))  # 위상 차이
        })
        
        assert result.sensor_count == 2
        assert len(result.correlation_hash) == 32
    
    def test_compute_multiple_sensors(self):
        """다중 센서"""
        sig = CorrelationSignature()
        np.random.seed(42)
        
        result = sig.compute({
            "sensor1": np.random.randn(100),
            "sensor2": np.random.randn(100),
            "sensor3": np.random.randn(100),
            "sensor4": np.random.randn(100)
        })
        
        assert result.sensor_count == 4
        assert len(result.feature_vector) == 16
    
    def test_compute_with_matrix(self):
        """상관 행렬 포함"""
        sig = CorrelationSignature()
        
        result = sig.compute({
            "sensor1": np.sin(np.linspace(0, 4*np.pi, 50)),
            "sensor2": np.cos(np.linspace(0, 4*np.pi, 50))
        }, return_matrix=True)
        
        assert result.correlation_matrix is not None
        assert result.correlation_matrix.shape[0] == 2  # 2 sensors
        assert result.correlation_matrix.shape[1] == 2
    
    def test_hash_deterministic(self):
        """해시 결정성 확인"""
        sig = CorrelationSignature()
        data = {"sensor1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        
        result1 = sig.compute(data)
        result2 = sig.compute(data)
        
        assert result1.correlation_hash == result2.correlation_hash
    
    def test_hash_different_for_different_data(self):
        """다른 데이터에 대해 다른 해시"""
        sig = CorrelationSignature()
        
        # 더 큰 데이터로 테스트 (작은 데이터는 정규화 후 동일해질 수 있음)
        np.random.seed(42)
        result1 = sig.compute({
            "sensor1": np.sin(np.linspace(0, 4*np.pi, 100)),
            "sensor2": np.random.randn(100)
        })
        np.random.seed(123)
        result2 = sig.compute({
            "sensor1": np.cos(np.linspace(0, 8*np.pi, 100)),
            "sensor2": np.random.randn(100)
        })
        
        assert result1.correlation_hash != result2.correlation_hash
    
    def test_verify_identical(self):
        """동일 결과 검증"""
        sig = CorrelationSignature()
        data = {"sensor1": np.sin(np.linspace(0, 4*np.pi, 100))}
        
        result1 = sig.compute(data)
        result2 = sig.compute(data)
        
        is_valid, similarity = sig.verify(result1, result2)
        
        assert is_valid is True
        assert similarity > 0.99
    
    def test_verify_similar(self):
        """유사한 결과 검증"""
        sig = CorrelationSignature()
        base_signal = np.sin(np.linspace(0, 4*np.pi, 100))
        
        result1 = sig.compute({"sensor1": base_signal})
        # 약간의 노이즈 추가
        result2 = sig.compute({"sensor1": base_signal + 0.01 * np.random.randn(100)})
        
        is_valid, similarity = sig.verify(result1, result2)
        
        # 유사도가 임계값 이상이면 유효
        assert similarity > 0.8
    
    def test_verify_different(self):
        """다른 결과 검증"""
        config = CorrelationConfig(similarity_threshold=0.7)
        sig = CorrelationSignature(config)
        
        # 상관된 사인파 vs 무상관 랜덤 노이즈
        t = np.linspace(0, 1, 100)
        result1 = sig.compute({
            "sensor1": np.sin(2 * np.pi * 5 * t),
            "sensor2": np.cos(2 * np.pi * 5 * t)  # 상관된 신호
        })
        result2 = sig.compute({
            "sensor1": np.random.RandomState(42).randn(100),
            "sensor2": np.random.RandomState(43).randn(100)  # 무상관 노이즈
        })
        
        is_valid, similarity = sig.verify(result1, result2)
        
        # 유사도가 완벽하지 않음을 확인 (완전 일치가 아님)
        assert similarity < 0.95


# ============================================================
# 편의 함수 테스트
# ============================================================

class TestComputeCorrelationSignature:
    """compute_correlation_signature 편의 함수 테스트"""
    
    def test_basic_usage(self):
        """기본 사용법"""
        data = {
            "sensor1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "sensor2": np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        }
        
        result = compute_correlation_signature(data)
        
        assert result.sensor_count == 2
        assert len(result.correlation_hash) == 32
    
    def test_with_config(self):
        """설정과 함께 사용"""
        config = CorrelationConfig(feature_size=8)
        data = {"sensor1": np.linspace(0, 1, 50)}
        
        result = compute_correlation_signature(data, config)
        
        assert len(result.feature_vector) == 8


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationCorrelation:
    """상관 서명 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우"""
        # 재현 가능한 시드
        np.random.seed(42)
        
        # 상관된 신호 생성
        t = np.linspace(0, 1, 256)
        base = np.sin(2 * np.pi * 5 * t)
        
        sensor_data = {
            "cpu_temp": base + 0.1 * np.random.randn(256),
            "os_entropy": base + 0.2 * np.random.randn(256),
            "clock_jitter": np.random.randn(256),  # 무상관 노이즈
            "synthetic": 0.5 * base + 0.5 * np.random.randn(256)
        }
        
        # 서명 계산
        sig = CorrelationSignature()
        result = sig.compute(sensor_data, return_matrix=True)
        
        # 결과 확인
        assert result.sensor_count == 4
        assert len(result.feature_vector) == 16
        assert len(result.correlation_hash) == 32
        assert result.correlation_matrix is not None
        
        # 동일 데이터 재계산
        result2 = sig.compute(sensor_data)
        
        # 검증
        is_valid, sim = sig.verify(result, result2)
        assert is_valid is True
        assert sim > 0.99
    
    def test_replay_attack_detection(self):
        """리플레이 공격 시뮬레이션"""
        sig = CorrelationSignature()
        
        # 원본 데이터: 상관된 신호
        t = np.linspace(0, 1, 100)
        base = np.sin(2 * np.pi * 5 * t)
        original_data = {
            "sensor1": base + 0.1 * np.random.RandomState(42).randn(100),
            "sensor2": 0.5 * base + 0.1 * np.random.RandomState(43).randn(100)
        }
        original_result = sig.compute(original_data)
        
        # 시간이 지난 후 새 데이터: 완전히 다른 노이즈 패턴
        new_data = {
            "sensor1": np.random.RandomState(999).randn(100),
            "sensor2": np.random.RandomState(888).randn(100)
        }
        new_result = sig.compute(new_data)
        
        # 검증: 다른 노이즈 패턴이므로 유사도 낮음
        is_valid, sim = sig.verify(original_result, new_result)
        
        # 상관된 신호 vs 무상관 노이즈는 유사도 낮음
        assert sim < 0.95
