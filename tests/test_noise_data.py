"""
NoiseChain 노이즈 데이터 모델 테스트

SensorType, Sample, TimeSeries 클래스의 단위 테스트입니다.
"""

import time

import numpy as np
import pytest

from noisechain.models.noise_data import Sample, SensorType, TimeSeries


class TestSensorType:
    """SensorType 열거형 테스트"""
    
    def test_sensor_types_exist(self):
        """모든 센서 타입이 정의되어 있는지 확인"""
        assert SensorType.CPU_TEMP is not None
        assert SensorType.OS_ENTROPY is not None
        assert SensorType.CLOCK_JITTER is not None
        assert SensorType.SYNTHETIC_NOISE is not None
    
    def test_value_range_cpu_temp(self):
        """CPU 온도 범위 확인"""
        min_val, max_val = SensorType.CPU_TEMP.value_range
        assert min_val == 30.0
        assert max_val == 100.0
    
    def test_value_range_os_entropy(self):
        """OS 엔트로피 범위 확인"""
        min_val, max_val = SensorType.OS_ENTROPY.value_range
        assert min_val == 0.0
        assert max_val == 255.0
    
    def test_value_range_clock_jitter(self):
        """클럭 지터 범위 확인"""
        min_val, max_val = SensorType.CLOCK_JITTER.value_range
        assert min_val == -1000.0
        assert max_val == 1000.0
    
    def test_value_range_synthetic_noise(self):
        """합성 노이즈 범위 확인"""
        min_val, max_val = SensorType.SYNTHETIC_NOISE.value_range
        assert min_val == -1.0
        assert max_val == 1.0
    
    def test_unit(self):
        """단위 문자열 확인"""
        assert SensorType.CPU_TEMP.unit == "°C"
        assert SensorType.OS_ENTROPY.unit == "byte"
        assert SensorType.CLOCK_JITTER.unit == "ns"
        assert SensorType.SYNTHETIC_NOISE.unit == "normalized"


class TestSample:
    """Sample 클래스 테스트"""
    
    def test_create_with_auto_timestamp(self):
        """자동 타임스탬프로 샘플 생성"""
        values = {SensorType.CPU_TEMP: 45.0}
        sample = Sample.create(values)
        
        assert sample.timestamp_ns > 0
        assert sample.get(SensorType.CPU_TEMP) == 45.0
    
    def test_create_with_explicit_timestamp(self):
        """명시적 타임스탬프로 샘플 생성"""
        ts = 1234567890_000000000
        values = {SensorType.CPU_TEMP: 50.0}
        sample = Sample.create(values, timestamp_ns=ts)
        
        assert sample.timestamp_ns == ts
    
    def test_get_existing_value(self):
        """존재하는 센서 값 조회"""
        sample = Sample.create({
            SensorType.CPU_TEMP: 55.0,
            SensorType.OS_ENTROPY: 128.0
        })
        
        assert sample.get(SensorType.CPU_TEMP) == 55.0
        assert sample.get(SensorType.OS_ENTROPY) == 128.0
    
    def test_get_missing_value_with_default(self):
        """없는 센서 값 조회 시 기본값 반환"""
        sample = Sample.create({SensorType.CPU_TEMP: 55.0})
        
        assert sample.get(SensorType.CLOCK_JITTER) == 0.0
        assert sample.get(SensorType.CLOCK_JITTER, default=-999.0) == -999.0
    
    def test_has_sensor(self):
        """센서 값 존재 여부 확인"""
        sample = Sample.create({SensorType.CPU_TEMP: 55.0})
        
        assert sample.has(SensorType.CPU_TEMP) is True
        assert sample.has(SensorType.CLOCK_JITTER) is False
    
    def test_sensor_count(self):
        """센서 개수 확인"""
        sample = Sample.create({
            SensorType.CPU_TEMP: 55.0,
            SensorType.OS_ENTROPY: 128.0,
            SensorType.CLOCK_JITTER: 5.0
        })
        
        assert sample.sensor_count == 3
    
    def test_validate_valid_sample(self):
        """유효한 샘플 검증"""
        sample = Sample.create({
            SensorType.CPU_TEMP: 50.0,  # 범위 내
            SensorType.OS_ENTROPY: 100.0  # 범위 내
        })
        
        is_valid, errors = sample.validate()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_invalid_range(self):
        """범위 벗어난 값 검증"""
        sample = Sample.create({
            SensorType.CPU_TEMP: 150.0,  # 범위 초과 (max: 100)
        })
        
        is_valid, errors = sample.validate()
        assert is_valid is False
        assert len(errors) == 1
        assert "CPU_TEMP" in errors[0]
    
    def test_validate_invalid_timestamp(self):
        """잘못된 타임스탬프 검증"""
        sample = Sample(timestamp_ns=-1, values={SensorType.CPU_TEMP: 50.0})
        
        is_valid, errors = sample.validate()
        assert is_valid is False
        assert "타임스탬프" in errors[0]


class TestTimeSeries:
    """TimeSeries 클래스 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        ts = TimeSeries()
        
        assert ts.sampling_rate_hz == 100
        assert ts.window_size == 256
        assert ts.max_samples == 1024
        assert ts.count == 0
    
    def test_add_sample(self):
        """샘플 추가"""
        ts = TimeSeries()
        sample = Sample.create({SensorType.CPU_TEMP: 45.0})
        
        ts.add_sample(sample)
        
        assert ts.count == 1
    
    def test_circular_buffer(self):
        """순환 버퍼 동작 확인"""
        ts = TimeSeries(max_samples=3)
        
        for i in range(5):
            sample = Sample.create({SensorType.CPU_TEMP: float(i)})
            ts.add_sample(sample)
        
        # 최대 3개만 유지
        assert ts.count == 3
        # 최근 3개 (2, 3, 4) 유지
        window = ts.get_window(3)
        assert window[0].get(SensorType.CPU_TEMP) == 2.0
        assert window[2].get(SensorType.CPU_TEMP) == 4.0
    
    def test_get_window_default_size(self):
        """기본 윈도우 크기로 조회"""
        ts = TimeSeries(window_size=3)
        
        for i in range(5):
            sample = Sample.create({SensorType.CPU_TEMP: float(i)})
            ts.add_sample(sample)
        
        window = ts.get_window()
        assert len(window) == 3
    
    def test_get_window_custom_size(self):
        """사용자 정의 윈도우 크기로 조회"""
        ts = TimeSeries()
        
        for i in range(10):
            sample = Sample.create({SensorType.CPU_TEMP: float(i)})
            ts.add_sample(sample)
        
        window = ts.get_window(size=5)
        assert len(window) == 5
    
    def test_get_window_insufficient_samples(self):
        """샘플 부족 시 있는 만큼 반환"""
        ts = TimeSeries(window_size=10)
        
        for i in range(3):
            sample = Sample.create({SensorType.CPU_TEMP: float(i)})
            ts.add_sample(sample)
        
        window = ts.get_window()
        assert len(window) == 3  # 요청 10개, 실제 3개
    
    def test_to_numpy_single_sensor(self):
        """단일 센서 numpy 변환"""
        ts = TimeSeries()
        
        for i in range(5):
            sample = Sample.create({SensorType.CPU_TEMP: float(i * 10)})
            ts.add_sample(sample)
        
        arrays = ts.to_numpy(sensor_types=[SensorType.CPU_TEMP], size=5)
        
        assert SensorType.CPU_TEMP in arrays
        np.testing.assert_array_equal(
            arrays[SensorType.CPU_TEMP], 
            np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        )
    
    def test_to_numpy_multiple_sensors(self):
        """다중 센서 numpy 변환"""
        ts = TimeSeries()
        
        for i in range(3):
            sample = Sample.create({
                SensorType.CPU_TEMP: float(i),
                SensorType.OS_ENTROPY: float(i * 10)
            })
            ts.add_sample(sample)
        
        arrays = ts.to_numpy(size=3)
        
        assert len(arrays) == 2
        assert SensorType.CPU_TEMP in arrays
        assert SensorType.OS_ENTROPY in arrays
    
    def test_to_numpy_empty(self):
        """빈 시계열 numpy 변환"""
        ts = TimeSeries()
        
        arrays = ts.to_numpy()
        assert arrays == {}
    
    def test_clear(self):
        """샘플 전체 삭제"""
        ts = TimeSeries()
        
        for i in range(5):
            sample = Sample.create({SensorType.CPU_TEMP: float(i)})
            ts.add_sample(sample)
        
        ts.clear()
        assert ts.count == 0
    
    def test_is_full(self):
        """버퍼 가득 참 여부"""
        ts = TimeSeries(max_samples=3)
        
        assert ts.is_full is False
        
        for i in range(3):
            ts.add_sample(Sample.create({SensorType.CPU_TEMP: float(i)}))
        
        assert ts.is_full is True
    
    def test_has_enough_samples(self):
        """윈도우 크기 충분 여부"""
        ts = TimeSeries(window_size=5)
        
        for i in range(3):
            ts.add_sample(Sample.create({SensorType.CPU_TEMP: float(i)}))
        
        assert ts.has_enough_samples is False
        
        for i in range(3):
            ts.add_sample(Sample.create({SensorType.CPU_TEMP: float(i)}))
        
        assert ts.has_enough_samples is True
    
    def test_time_span(self):
        """시간 범위 계산"""
        ts = TimeSeries()
        
        # 빈 시계열
        assert ts.time_span_ns == 0
        
        # 1개 샘플
        ts.add_sample(Sample.create({SensorType.CPU_TEMP: 50.0}, timestamp_ns=1_000_000_000))
        assert ts.time_span_ns == 0
        
        # 2개 샘플 (1초 차이)
        ts.add_sample(Sample.create({SensorType.CPU_TEMP: 50.0}, timestamp_ns=2_000_000_000))
        assert ts.time_span_ns == 1_000_000_000
        assert ts.time_span_ms == 1000.0


class TestIntegration:
    """통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 샘플 생성 → 시계열 추가 → numpy 변환"""
        # 시계열 생성
        ts = TimeSeries(sampling_rate_hz=100, window_size=10)
        
        # 샘플 추가
        for i in range(20):
            sample = Sample.create({
                SensorType.CPU_TEMP: 50.0 + i * 0.5,
                SensorType.OS_ENTROPY: float(i % 256),
                SensorType.SYNTHETIC_NOISE: np.sin(i * 0.1)
            })
            
            # 유효성 검사
            is_valid, _ = sample.validate()
            assert is_valid
            
            ts.add_sample(sample)
        
        # 윈도우 추출
        assert ts.has_enough_samples is True
        window = ts.get_window()
        assert len(window) == 10
        
        # numpy 변환
        arrays = ts.to_numpy()
        assert len(arrays) == 3
        assert arrays[SensorType.CPU_TEMP].shape == (10,)
