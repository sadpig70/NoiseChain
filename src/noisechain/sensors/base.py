"""
NoiseChain 센서 드라이버 베이스 클래스

모든 센서 드라이버가 구현해야 하는 추상 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod

from noisechain.models.noise_data import SensorType


class BaseSensorDriver(ABC):
    """
    센서 드라이버 추상 베이스 클래스
    
    모든 센서 드라이버는 이 클래스를 상속받아 구현해야 합니다.
    
    Attributes:
        _sensor_type: 센서 타입
        _enabled: 활성화 여부
    """
    
    def __init__(self, sensor_type: SensorType):
        """
        Args:
            sensor_type: 센서 타입 열거형
        """
        self._sensor_type = sensor_type
        self._enabled = True
    
    @property
    def sensor_type(self) -> SensorType:
        """센서 타입 반환"""
        return self._sensor_type
    
    @property
    def enabled(self) -> bool:
        """활성화 여부"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """활성화 여부 설정"""
        self._enabled = value
    
    @abstractmethod
    def read(self) -> float:
        """
        센서 값 읽기 (구현 필수)
        
        Returns:
            센서 측정값 (센서 타입별 유효 범위 내)
        
        Raises:
            SensorReadError: 센서 읽기 실패 시
        """
        pass
    
    def read_safe(self, default: float | None = None) -> float:
        """
        안전한 센서 값 읽기 (예외 발생 시 기본값 반환)
        
        Args:
            default: 실패 시 반환할 기본값. None이면 센서 범위 중간값 사용.
        
        Returns:
            센서 측정값 또는 기본값
        """
        if default is None:
            min_val, max_val = self._sensor_type.value_range
            default = (min_val + max_val) / 2
        
        try:
            return self.read()
        except Exception:
            return default
    
    def validate_value(self, value: float) -> bool:
        """
        값이 유효 범위 내인지 확인
        
        Args:
            value: 검사할 값
        
        Returns:
            유효 여부
        """
        min_val, max_val = self._sensor_type.value_range
        return min_val <= value <= max_val
    
    def clamp_value(self, value: float) -> float:
        """
        값을 유효 범위로 클램핑
        
        Args:
            value: 클램핑할 값
        
        Returns:
            범위 내로 조정된 값
        """
        min_val, max_val = self._sensor_type.value_range
        return max(min_val, min(max_val, value))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self._sensor_type.name}, enabled={self._enabled})"


class SensorReadError(Exception):
    """센서 읽기 실패 예외"""
    
    def __init__(self, sensor_type: SensorType, message: str):
        self.sensor_type = sensor_type
        self.message = message
        super().__init__(f"[{sensor_type.name}] {message}")
