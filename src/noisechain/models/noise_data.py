"""
NoiseChain 노이즈 데이터 모델

가상 센서에서 수집되는 노이즈 데이터의 구조를 정의합니다.

Classes:
    SensorType: 센서 타입 열거형
    Sample: 단일 샘플 데이터
    TimeSeries: 시계열 버퍼
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class SensorType(Enum):
    """
    가상 센서 타입 열거형
    
    각 센서 타입은 고유한 데이터 범위와 특성을 가집니다.
    
    Attributes:
        CPU_TEMP: CPU 온도 (30.0 ~ 100.0 °C)
        OS_ENTROPY: OS 엔트로피 (0 ~ 255, 바이트 값)
        CLOCK_JITTER: 클럭 지터 (-1000 ~ 1000 ns)
        SYNTHETIC_NOISE: 합성 노이즈 (-1.0 ~ 1.0, 정규화)
    """
    CPU_TEMP = auto()
    OS_ENTROPY = auto()
    CLOCK_JITTER = auto()
    SYNTHETIC_NOISE = auto()
    
    @property
    def value_range(self) -> tuple[float, float]:
        """센서 타입별 유효 값 범위 반환"""
        ranges = {
            SensorType.CPU_TEMP: (30.0, 100.0),
            SensorType.OS_ENTROPY: (0.0, 255.0),
            SensorType.CLOCK_JITTER: (-1000.0, 1000.0),
            SensorType.SYNTHETIC_NOISE: (-1.0, 1.0),
        }
        return ranges[self]
    
    @property
    def unit(self) -> str:
        """센서 타입별 단위 반환"""
        units = {
            SensorType.CPU_TEMP: "°C",
            SensorType.OS_ENTROPY: "byte",
            SensorType.CLOCK_JITTER: "ns",
            SensorType.SYNTHETIC_NOISE: "normalized",
        }
        return units[self]


@dataclass
class Sample:
    """
    단일 샘플 데이터
    
    특정 시점에 수집된 다중 센서 값을 담는 컨테이너입니다.
    
    Attributes:
        timestamp_ns: 나노초 단위 타임스탬프 (Unix epoch 기준)
        values: 센서 타입별 측정값 딕셔너리
    
    Example:
        >>> sample = Sample.create({
        ...     SensorType.CPU_TEMP: 45.2,
        ...     SensorType.OS_ENTROPY: 127.0
        ... })
        >>> sample.get(SensorType.CPU_TEMP)
        45.2
    """
    timestamp_ns: int
    values: dict[SensorType, float] = field(default_factory=dict)
    
    @classmethod
    def create(cls, values: dict[SensorType, float], timestamp_ns: Optional[int] = None) -> Sample:
        """
        새 샘플 생성
        
        Args:
            values: 센서 타입별 측정값
            timestamp_ns: 타임스탬프 (None이면 현재 시간 사용)
        
        Returns:
            생성된 Sample 객체
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        return cls(timestamp_ns=timestamp_ns, values=values)
    
    def get(self, sensor_type: SensorType, default: float = 0.0) -> float:
        """특정 센서 값 조회"""
        return self.values.get(sensor_type, default)
    
    def has(self, sensor_type: SensorType) -> bool:
        """특정 센서 값 존재 여부 확인"""
        return sensor_type in self.values
    
    @property
    def sensor_count(self) -> int:
        """포함된 센서 개수"""
        return len(self.values)
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        샘플 유효성 검사
        
        Returns:
            (유효 여부, 에러 메시지 목록)
        """
        errors = []
        
        # 타임스탬프 검사
        if self.timestamp_ns <= 0:
            errors.append("타임스탬프는 양수여야 합니다")
        
        # 값 범위 검사
        for sensor_type, value in self.values.items():
            min_val, max_val = sensor_type.value_range
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{sensor_type.name}: {value}는 유효 범위 [{min_val}, {max_val}]를 벗어남"
                )
        
        return len(errors) == 0, errors


@dataclass
class TimeSeries:
    """
    시계열 버퍼
    
    연속적인 샘플을 저장하고 윈도우 추출, numpy 변환 등을 지원합니다.
    
    Attributes:
        samples: 샘플 리스트
        sampling_rate_hz: 샘플링 레이트 (Hz)
        window_size: 기본 윈도우 크기 (샘플 수)
        max_samples: 최대 저장 샘플 수 (순환 버퍼)
    
    Example:
        >>> ts = TimeSeries(sampling_rate_hz=100, window_size=256)
        >>> ts.add_sample(sample)
        >>> window = ts.get_window()
    """
    sampling_rate_hz: int = 100
    window_size: int = 256
    max_samples: int = 1024
    samples: list[Sample] = field(default_factory=list)
    
    def add_sample(self, sample: Sample) -> None:
        """
        샘플 추가
        
        max_samples를 초과하면 가장 오래된 샘플 제거 (순환 버퍼)
        """
        self.samples.append(sample)
        
        # 순환 버퍼: 최대 개수 초과 시 오래된 샘플 제거
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]
    
    def add_samples(self, samples: list[Sample]) -> None:
        """다수 샘플 일괄 추가"""
        for sample in samples:
            self.add_sample(sample)
    
    def get_window(self, size: Optional[int] = None) -> list[Sample]:
        """
        최근 N개 샘플 윈도우 반환
        
        Args:
            size: 윈도우 크기 (None이면 기본값 사용)
        
        Returns:
            최근 샘플 리스트 (샘플 부족 시 있는 만큼 반환)
        """
        if size is None:
            size = self.window_size
        return self.samples[-size:]
    
    def to_numpy(
        self, 
        sensor_types: Optional[list[SensorType]] = None,
        size: Optional[int] = None
    ) -> dict[SensorType, np.ndarray]:
        """
        numpy 배열로 변환
        
        Args:
            sensor_types: 추출할 센서 타입 (None이면 모든 타입)
            size: 윈도우 크기 (None이면 기본값 사용)
        
        Returns:
            센서 타입별 numpy 배열 딕셔너리
        """
        window = self.get_window(size)
        
        if not window:
            return {}
        
        # 센서 타입 결정
        if sensor_types is None:
            # 첫 샘플의 센서 타입 사용
            sensor_types = list(window[0].values.keys())
        
        # 각 센서 타입별 배열 생성
        result = {}
        for sensor_type in sensor_types:
            values = [sample.get(sensor_type) for sample in window]
            result[sensor_type] = np.array(values, dtype=np.float64)
        
        return result
    
    def clear(self) -> None:
        """모든 샘플 제거"""
        self.samples = []
    
    @property
    def count(self) -> int:
        """현재 저장된 샘플 수"""
        return len(self.samples)
    
    @property
    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 여부"""
        return len(self.samples) >= self.max_samples
    
    @property
    def has_enough_samples(self) -> bool:
        """윈도우 크기만큼 샘플이 있는지 여부"""
        return len(self.samples) >= self.window_size
    
    @property
    def time_span_ns(self) -> int:
        """현재 샘플의 시간 범위 (나노초)"""
        if len(self.samples) < 2:
            return 0
        return self.samples[-1].timestamp_ns - self.samples[0].timestamp_ns
    
    @property
    def time_span_ms(self) -> float:
        """현재 샘플의 시간 범위 (밀리초)"""
        return self.time_span_ns / 1_000_000
