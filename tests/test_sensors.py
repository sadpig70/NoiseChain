"""
NoiseChain 센서 드라이버 테스트

BaseSensorDriver, 가상 드라이버, SensorHub 단위 테스트입니다.
"""

import time

import numpy as np
import pytest

from noisechain.models.noise_data import SensorType
from noisechain.sensors.base import BaseSensorDriver, SensorReadError
from noisechain.sensors.hub import HubConfig, SensorHub
from noisechain.sensors.virtual_drivers import (
    ClockJitterDriver,
    CPUTempDriver,
    DriverConfig,
    OSEntropyDriver,
    SyntheticNoiseDriver,
    create_default_drivers,
)


# ============================================================
# BaseSensorDriver 테스트
# ============================================================

class ConcreteDriver(BaseSensorDriver):
    """테스트용 구체 드라이버"""
    
    def __init__(self, sensor_type: SensorType, value: float = 50.0):
        super().__init__(sensor_type)
        self._value = value
        self._should_fail = False
    
    def read(self) -> float:
        if self._should_fail:
            raise SensorReadError(self._sensor_type, "테스트 실패")
        return self._value


class TestBaseSensorDriver:
    """BaseSensorDriver 테스트"""
    
    def test_sensor_type_property(self):
        """센서 타입 속성 확인"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert driver.sensor_type == SensorType.CPU_TEMP
    
    def test_enabled_default(self):
        """기본 활성화 상태"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert driver.enabled is True
    
    def test_enabled_setter(self):
        """활성화 상태 변경"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        driver.enabled = False
        assert driver.enabled is False
    
    def test_read(self):
        """값 읽기"""
        driver = ConcreteDriver(SensorType.CPU_TEMP, value=65.0)
        assert driver.read() == 65.0
    
    def test_read_safe_success(self):
        """안전 읽기 성공"""
        driver = ConcreteDriver(SensorType.CPU_TEMP, value=70.0)
        assert driver.read_safe() == 70.0
    
    def test_read_safe_failure_default(self):
        """안전 읽기 실패 시 기본값"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        driver._should_fail = True
        # 기본값: (30 + 100) / 2 = 65
        assert driver.read_safe() == 65.0
    
    def test_read_safe_failure_custom_default(self):
        """안전 읽기 실패 시 사용자 정의 기본값"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        driver._should_fail = True
        assert driver.read_safe(default=99.0) == 99.0
    
    def test_validate_value_in_range(self):
        """범위 내 값 검증"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert driver.validate_value(50.0) is True
        assert driver.validate_value(30.0) is True
        assert driver.validate_value(100.0) is True
    
    def test_validate_value_out_of_range(self):
        """범위 외 값 검증"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert driver.validate_value(29.9) is False
        assert driver.validate_value(100.1) is False
    
    def test_clamp_value(self):
        """값 클램핑"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert driver.clamp_value(50.0) == 50.0
        assert driver.clamp_value(25.0) == 30.0  # min
        assert driver.clamp_value(110.0) == 100.0  # max
    
    def test_repr(self):
        """문자열 표현"""
        driver = ConcreteDriver(SensorType.CPU_TEMP)
        assert "ConcreteDriver" in repr(driver)
        assert "CPU_TEMP" in repr(driver)


class TestSensorReadError:
    """SensorReadError 테스트"""
    
    def test_error_message(self):
        """에러 메시지 형식"""
        error = SensorReadError(SensorType.CPU_TEMP, "테스트 메시지")
        assert "[CPU_TEMP]" in str(error)
        assert "테스트 메시지" in str(error)


# ============================================================
# 가상 드라이버 테스트
# ============================================================

class TestCPUTempDriver:
    """CPUTempDriver 테스트"""
    
    def test_sensor_type(self):
        """센서 타입 확인"""
        driver = CPUTempDriver()
        assert driver.sensor_type == SensorType.CPU_TEMP
    
    def test_read_returns_value_in_range(self):
        """읽기 값이 범위 내인지 확인"""
        driver = CPUTempDriver(base_temp=50.0, variation=10.0)
        for _ in range(10):
            value = driver.read()
            assert 30.0 <= value <= 100.0
    
    def test_simulation_mode(self):
        """시뮬레이션 모드 동작"""
        driver = CPUTempDriver(base_temp=60.0, variation=5.0)
        driver._use_simulation = True
        values = [driver.read() for _ in range(100)]
        # 평균이 기준 온도 근처인지 확인
        assert 55.0 <= np.mean(values) <= 65.0


class TestOSEntropyDriver:
    """OSEntropyDriver 테스트"""
    
    def test_sensor_type(self):
        """센서 타입 확인"""
        driver = OSEntropyDriver()
        assert driver.sensor_type == SensorType.OS_ENTROPY
    
    def test_read_returns_value_in_range(self):
        """읽기 값이 범위 내 (0~255)"""
        driver = OSEntropyDriver(bytes_to_read=32)
        for _ in range(10):
            value = driver.read()
            assert 0.0 <= value <= 255.0
    
    def test_read_is_random(self):
        """읽기 값이 랜덤인지 확인"""
        driver = OSEntropyDriver(bytes_to_read=32)
        values = [driver.read() for _ in range(100)]
        # 모든 값이 동일하지 않음
        assert len(set(values)) > 1


class TestClockJitterDriver:
    """ClockJitterDriver 테스트"""
    
    def test_sensor_type(self):
        """센서 타입 확인"""
        driver = ClockJitterDriver()
        assert driver.sensor_type == SensorType.CLOCK_JITTER
    
    def test_read_returns_value_in_range(self):
        """읽기 값이 범위 내 (-1000~1000 ns)"""
        driver = ClockJitterDriver(samples=10)
        for _ in range(10):
            value = driver.read()
            assert -1000.0 <= value <= 1000.0


class TestSyntheticNoiseDriver:
    """SyntheticNoiseDriver 테스트"""
    
    def test_sensor_type(self):
        """센서 타입 확인"""
        driver = SyntheticNoiseDriver()
        assert driver.sensor_type == SensorType.SYNTHETIC_NOISE
    
    def test_read_returns_value_in_range(self):
        """읽기 값이 범위 내 (-1~1)"""
        driver = SyntheticNoiseDriver(mean=0.0, std=0.3)
        for _ in range(100):
            value = driver.read()
            assert -1.0 <= value <= 1.0
    
    def test_reproducibility_with_seed(self):
        """시드로 재현 가능"""
        driver1 = SyntheticNoiseDriver(seed=42)
        driver2 = SyntheticNoiseDriver(seed=42)
        
        values1 = [driver1.read() for _ in range(10)]
        values2 = [driver2.read() for _ in range(10)]
        
        np.testing.assert_array_almost_equal(values1, values2)
    
    def test_set_seed(self):
        """시드 재설정"""
        driver = SyntheticNoiseDriver(seed=42)
        values1 = [driver.read() for _ in range(5)]
        
        driver.set_seed(42)
        values2 = [driver.read() for _ in range(5)]
        
        np.testing.assert_array_almost_equal(values1, values2)


class TestCreateDefaultDrivers:
    """create_default_drivers 팩토리 함수 테스트"""
    
    def test_creates_all_drivers(self):
        """모든 드라이버 생성 확인"""
        drivers = create_default_drivers()
        
        assert SensorType.CPU_TEMP in drivers
        assert SensorType.OS_ENTROPY in drivers
        assert SensorType.CLOCK_JITTER in drivers
        assert SensorType.SYNTHETIC_NOISE in drivers
    
    def test_custom_config(self):
        """사용자 정의 설정 적용"""
        config = DriverConfig(
            cpu_temp_base=70.0,
            noise_seed=123
        )
        drivers = create_default_drivers(config)
        
        cpu_driver = drivers[SensorType.CPU_TEMP]
        assert cpu_driver.base_temp == 70.0


# ============================================================
# SensorHub 테스트
# ============================================================

class TestSensorHub:
    """SensorHub 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        hub = SensorHub()
        
        assert len(hub.drivers) == 4
        assert hub.sample_count == 0
        assert hub.is_running is False
    
    def test_custom_config(self):
        """사용자 정의 설정"""
        config = HubConfig(sampling_rate_hz=50, window_size=128)
        hub = SensorHub(config=config)
        
        assert hub.config.sampling_rate_hz == 50
        assert hub.config.window_size == 128
    
    def test_register_driver(self):
        """드라이버 등록"""
        hub = SensorHub()
        custom_driver = SyntheticNoiseDriver(mean=0.5)
        
        hub.register_driver(custom_driver)
        
        driver = hub.get_driver(SensorType.SYNTHETIC_NOISE)
        assert driver.mean == 0.5
    
    def test_unregister_driver(self):
        """드라이버 등록 해제"""
        hub = SensorHub()
        
        result = hub.unregister_driver(SensorType.CPU_TEMP)
        assert result is True
        assert hub.get_driver(SensorType.CPU_TEMP) is None
    
    def test_collect_sample(self):
        """단일 샘플 수집"""
        hub = SensorHub()
        
        sample = hub.collect_sample()
        
        assert sample is not None
        assert sample.sensor_count == 4
        assert hub.sample_count == 1
    
    def test_collect_samples(self):
        """다수 샘플 수집"""
        hub = SensorHub()
        
        samples = hub.collect_samples(10)
        
        assert len(samples) == 10
        assert hub.sample_count == 10
    
    def test_get_window(self):
        """윈도우 조회"""
        hub = SensorHub()
        hub.collect_samples(20)
        
        window = hub.get_window(10)
        
        assert len(window) == 10
    
    def test_get_numpy_window(self):
        """numpy 윈도우 조회"""
        hub = SensorHub()
        hub.collect_samples(10)
        
        arrays = hub.get_numpy_window(size=10)
        
        assert len(arrays) == 4
        assert arrays[SensorType.CPU_TEMP].shape == (10,)
    
    def test_clear(self):
        """버퍼 초기화"""
        hub = SensorHub()
        hub.collect_samples(10)
        
        hub.clear()
        
        assert hub.sample_count == 0
    
    def test_has_enough_samples(self):
        """충분한 샘플 여부"""
        config = HubConfig(window_size=10)
        hub = SensorHub(config=config)
        
        assert hub.has_enough_samples is False
        
        hub.collect_samples(10)
        
        assert hub.has_enough_samples is True
    
    def test_registered_sensors(self):
        """등록된 센서 목록"""
        hub = SensorHub()
        
        sensors = hub.registered_sensors
        
        assert len(sensors) == 4
        assert SensorType.CPU_TEMP in sensors
    
    def test_context_manager(self):
        """컨텍스트 매니저 동작"""
        with SensorHub() as hub:
            hub.collect_samples(5)
            assert hub.sample_count == 5
        
        # 종료 후 running 상태 확인
        assert hub.is_running is False
    
    def test_continuous_collection(self):
        """연속 수집 시작/중지"""
        config = HubConfig(sampling_rate_hz=100)
        hub = SensorHub(config=config)
        
        # 시작
        result = hub.start_collection()
        assert result is True
        assert hub.is_running is True
        
        # 잠시 대기
        time.sleep(0.1)
        
        # 중지
        result = hub.stop_collection()
        assert result is True
        assert hub.is_running is False
        
        # 샘플이 수집되었는지 확인
        assert hub.sample_count > 0
    
    def test_continuous_collection_with_callback(self):
        """콜백과 함께 연속 수집"""
        hub = SensorHub()
        collected = []
        
        def on_sample(sample):
            collected.append(sample)
        
        hub.start_collection(on_sample=on_sample)
        time.sleep(0.05)
        hub.stop_collection()
        
        assert len(collected) > 0
    
    def test_repr(self):
        """문자열 표현"""
        hub = SensorHub()
        hub.collect_samples(5)
        
        repr_str = repr(hub)
        
        assert "SensorHub" in repr_str
        assert "sensors=4" in repr_str
        assert "samples=5" in repr_str


class TestIntegrationSensors:
    """센서 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 허브 생성 → 샘플 수집 → numpy 변환"""
        # 재현 가능한 설정
        config = HubConfig(
            sampling_rate_hz=100,
            window_size=50,
            driver_config=DriverConfig(noise_seed=42)
        )
        
        with SensorHub(config=config) as hub:
            # 샘플 수집
            hub.collect_samples(100)
            
            assert hub.has_enough_samples is True
            
            # numpy 변환
            arrays = hub.get_numpy_window(size=50)
            
            # 모든 센서 데이터 존재
            assert len(arrays) == 4
            
            # 데이터 통계 확인
            cpu_temps = arrays[SensorType.CPU_TEMP]
            assert len(cpu_temps) == 50
            assert 30.0 <= np.mean(cpu_temps) <= 100.0
