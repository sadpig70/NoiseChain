"""
NoiseChain 가상 센서 드라이버

하드웨어 센서 없이 시뮬레이션용 노이즈 데이터를 생성하는 드라이버 모음입니다.

Classes:
    CPUTempDriver: CPU 온도 센서 (psutil 또는 랜덤)
    OSEntropyDriver: OS 엔트로피 (/dev/urandom 또는 os.urandom)
    ClockJitterDriver: 클럭 지터 (time.perf_counter_ns 변동)
    SyntheticNoiseDriver: 합성 가우시안 노이즈
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np

from noisechain.models.noise_data import SensorType
from noisechain.sensors.base import BaseSensorDriver, SensorReadError

# psutil은 플랫폼에 따라 CPU 온도 지원 여부가 다름
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class CPUTempDriver(BaseSensorDriver):
    """
    CPU 온도 센서 드라이버
    
    psutil을 통해 실제 CPU 온도를 읽거나, 
    사용 불가 시 시뮬레이션 값을 생성합니다.
    
    Attributes:
        base_temp: 기준 온도 (시뮬레이션용)
        variation: 변동 범위 (시뮬레이션용)
    """
    
    def __init__(self, base_temp: float = 50.0, variation: float = 10.0):
        """
        Args:
            base_temp: 기준 온도 (기본 50°C)
            variation: 변동 범위 (기본 ±10°C)
        """
        super().__init__(SensorType.CPU_TEMP)
        self.base_temp = base_temp
        self.variation = variation
        self._use_simulation = not self._check_real_sensor()
    
    def _check_real_sensor(self) -> bool:
        """실제 CPU 온도 센서 사용 가능 여부 확인"""
        if not HAS_PSUTIL:
            return False
        try:
            temps = psutil.sensors_temperatures()
            return bool(temps)
        except Exception:
            return False
    
    def _read_real(self) -> float:
        """실제 CPU 온도 읽기"""
        temps = psutil.sensors_temperatures()
        
        # 첫 번째 사용 가능한 온도 센서 값 반환
        for name, entries in temps.items():
            if entries:
                return entries[0].current
        
        raise SensorReadError(self._sensor_type, "온도 센서를 찾을 수 없음")
    
    def _read_simulation(self) -> float:
        """시뮬레이션 온도 생성"""
        # 가우시안 노이즈로 변동
        value = self.base_temp + np.random.normal(0, self.variation / 3)
        return self.clamp_value(value)
    
    def read(self) -> float:
        """CPU 온도 읽기"""
        if self._use_simulation:
            return self._read_simulation()
        
        try:
            return self._read_real()
        except Exception:
            # 실패 시 시뮬레이션으로 폴백
            return self._read_simulation()


class OSEntropyDriver(BaseSensorDriver):
    """
    OS 엔트로피 센서 드라이버
    
    운영체제의 엔트로피 소스(urandom)에서 바이트를 읽어
    0~255 범위의 값으로 변환합니다.
    
    Attributes:
        bytes_to_read: 읽을 바이트 수 (평균 계산용)
    """
    
    def __init__(self, bytes_to_read: int = 16):
        """
        Args:
            bytes_to_read: 읽을 바이트 수 (기본 16)
        """
        super().__init__(SensorType.OS_ENTROPY)
        self.bytes_to_read = bytes_to_read
    
    def read(self) -> float:
        """OS 엔트로피 읽기 (바이트 평균값)"""
        try:
            random_bytes = os.urandom(self.bytes_to_read)
            # 바이트 평균값 계산 (0~255)
            return float(sum(random_bytes) / len(random_bytes))
        except Exception as e:
            raise SensorReadError(self._sensor_type, f"urandom 읽기 실패: {e}")


class ClockJitterDriver(BaseSensorDriver):
    """
    클럭 지터 센서 드라이버
    
    고정밀 타이머의 연속 호출 간 시간 차이를 측정하여
    시스템 클럭 지터를 추정합니다.
    
    Attributes:
        samples: 측정 샘플 수
        _last_time: 마지막 측정 시간
    """
    
    def __init__(self, samples: int = 10):
        """
        Args:
            samples: 지터 측정용 샘플 수 (기본 10)
        """
        super().__init__(SensorType.CLOCK_JITTER)
        self.samples = samples
        self._last_time = time.perf_counter_ns()
    
    def read(self) -> float:
        """클럭 지터 측정 (나노초)"""
        # 여러 번 측정하여 표준편차 계산
        deltas = []
        for _ in range(self.samples):
            t1 = time.perf_counter_ns()
            t2 = time.perf_counter_ns()
            deltas.append(t2 - t1)
        
        # 평균 델타에서 기대값을 뺀 값 (지터 추정)
        mean_delta = np.mean(deltas)
        jitter = mean_delta - np.median(deltas)
        
        return self.clamp_value(jitter)


class SyntheticNoiseDriver(BaseSensorDriver):
    """
    합성 노이즈 드라이버
    
    가우시안 분포를 따르는 합성 노이즈를 생성합니다.
    테스트 및 시뮬레이션 용도로 사용됩니다.
    
    Attributes:
        mean: 평균
        std: 표준편차
        seed: 랜덤 시드 (재현성)
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.3, seed: int | None = None):
        """
        Args:
            mean: 노이즈 평균 (기본 0.0)
            std: 노이즈 표준편차 (기본 0.3)
            seed: 랜덤 시드 (None이면 비결정적)
        """
        super().__init__(SensorType.SYNTHETIC_NOISE)
        self.mean = mean
        self.std = std
        self._rng = np.random.default_rng(seed)
    
    def read(self) -> float:
        """합성 노이즈 생성"""
        value = self._rng.normal(self.mean, self.std)
        return self.clamp_value(value)
    
    def set_seed(self, seed: int) -> None:
        """랜덤 시드 재설정 (재현성 확보)"""
        self._rng = np.random.default_rng(seed)


@dataclass
class DriverConfig:
    """센서 드라이버 설정"""
    cpu_temp_base: float = 50.0
    cpu_temp_variation: float = 10.0
    entropy_bytes: int = 16
    jitter_samples: int = 10
    noise_mean: float = 0.0
    noise_std: float = 0.3
    noise_seed: int | None = None


def create_default_drivers(config: DriverConfig | None = None) -> dict[SensorType, BaseSensorDriver]:
    """
    기본 드라이버 세트 생성
    
    Args:
        config: 드라이버 설정 (None이면 기본값 사용)
    
    Returns:
        센서 타입별 드라이버 딕셔너리
    """
    if config is None:
        config = DriverConfig()
    
    return {
        SensorType.CPU_TEMP: CPUTempDriver(
            base_temp=config.cpu_temp_base,
            variation=config.cpu_temp_variation
        ),
        SensorType.OS_ENTROPY: OSEntropyDriver(
            bytes_to_read=config.entropy_bytes
        ),
        SensorType.CLOCK_JITTER: ClockJitterDriver(
            samples=config.jitter_samples
        ),
        SensorType.SYNTHETIC_NOISE: SyntheticNoiseDriver(
            mean=config.noise_mean,
            std=config.noise_std,
            seed=config.noise_seed
        ),
    }
