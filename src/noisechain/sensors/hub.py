"""
NoiseChain 센서 허브

다중 센서 드라이버를 통합 관리하고 동시 샘플링을 수행합니다.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from noisechain.models.noise_data import Sample, SensorType, TimeSeries
from noisechain.sensors.base import BaseSensorDriver
from noisechain.sensors.virtual_drivers import create_default_drivers, DriverConfig


@dataclass
class HubConfig:
    """센서 허브 설정"""
    sampling_rate_hz: int = 100      # 샘플링 레이트
    window_size: int = 256           # 윈도우 크기
    max_samples: int = 1024          # 최대 버퍼 크기
    driver_config: DriverConfig = field(default_factory=DriverConfig)


class SensorHub:
    """
    센서 허브
    
    여러 센서 드라이버를 등록하고, 동시에 샘플을 수집하여
    TimeSeries 버퍼에 저장합니다.
    
    Attributes:
        drivers: 센서 타입별 드라이버 딕셔너리
        time_series: 시계열 버퍼
        config: 허브 설정
    
    Example:
        >>> hub = SensorHub()
        >>> sample = hub.collect_sample()
        >>> hub.start_collection()
        >>> time.sleep(1)
        >>> hub.stop_collection()
        >>> arrays = hub.get_numpy_window()
    """
    
    def __init__(
        self, 
        config: HubConfig | None = None,
        drivers: dict[SensorType, BaseSensorDriver] | None = None
    ):
        """
        Args:
            config: 허브 설정 (None이면 기본값 사용)
            drivers: 사용할 드라이버 (None이면 기본 드라이버 생성)
        """
        self.config = config or HubConfig()
        
        # 드라이버 초기화
        if drivers is not None:
            self.drivers = drivers
        else:
            self.drivers = create_default_drivers(self.config.driver_config)
        
        # 시계열 버퍼 초기화
        self.time_series = TimeSeries(
            sampling_rate_hz=self.config.sampling_rate_hz,
            window_size=self.config.window_size,
            max_samples=self.config.max_samples
        )
        
        # 연속 수집 상태
        self._running = False
        self._collection_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        
        # 콜백
        self._on_sample: Callable[[Sample], None] | None = None
    
    def register_driver(self, driver: BaseSensorDriver) -> None:
        """
        드라이버 등록
        
        Args:
            driver: 등록할 센서 드라이버
        """
        with self._lock:
            self.drivers[driver.sensor_type] = driver
    
    def unregister_driver(self, sensor_type: SensorType) -> bool:
        """
        드라이버 등록 해제
        
        Args:
            sensor_type: 해제할 센서 타입
        
        Returns:
            해제 성공 여부
        """
        with self._lock:
            if sensor_type in self.drivers:
                del self.drivers[sensor_type]
                return True
            return False
    
    def get_driver(self, sensor_type: SensorType) -> BaseSensorDriver | None:
        """드라이버 조회"""
        return self.drivers.get(sensor_type)
    
    def get_enabled_drivers(self) -> list[BaseSensorDriver]:
        """활성화된 드라이버 목록"""
        return [d for d in self.drivers.values() if d.enabled]
    
    def collect_sample(self) -> Sample:
        """
        단일 샘플 수집
        
        모든 활성화된 센서에서 값을 읽어 Sample 객체 생성
        
        Returns:
            수집된 샘플
        """
        values = {}
        
        for sensor_type, driver in self.drivers.items():
            if driver.enabled:
                try:
                    values[sensor_type] = driver.read_safe()
                except Exception:
                    # 읽기 실패 시 기본값
                    min_val, max_val = sensor_type.value_range
                    values[sensor_type] = (min_val + max_val) / 2
        
        sample = Sample.create(values)
        
        # 버퍼에 추가
        with self._lock:
            self.time_series.add_sample(sample)
        
        # 콜백 호출
        if self._on_sample is not None:
            try:
                self._on_sample(sample)
            except Exception:
                pass  # 콜백 예외 무시
        
        return sample
    
    def collect_samples(self, count: int) -> list[Sample]:
        """
        다수 샘플 수집
        
        Args:
            count: 수집할 샘플 수
        
        Returns:
            수집된 샘플 리스트
        """
        return [self.collect_sample() for _ in range(count)]
    
    def start_collection(self, on_sample: Callable[[Sample], None] | None = None) -> bool:
        """
        연속 수집 시작
        
        Args:
            on_sample: 샘플 수집 시 호출될 콜백 (선택)
        
        Returns:
            시작 성공 여부
        """
        if self._running:
            return False
        
        self._on_sample = on_sample
        self._running = True
        
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        return True
    
    def stop_collection(self) -> bool:
        """
        연속 수집 중지
        
        Returns:
            중지 성공 여부
        """
        if not self._running:
            return False
        
        self._running = False
        
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=1.0)
            self._collection_thread = None
        
        return True
    
    def _collection_loop(self) -> None:
        """연속 수집 루프 (내부용)"""
        interval = 1.0 / self.config.sampling_rate_hz
        
        while self._running:
            start = time.perf_counter()
            
            self.collect_sample()
            
            # 정확한 간격 유지
            elapsed = time.perf_counter() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_window(self, size: int | None = None) -> list[Sample]:
        """
        현재 윈도우 반환
        
        Args:
            size: 윈도우 크기 (None이면 기본값)
        
        Returns:
            최근 샘플 리스트
        """
        with self._lock:
            return self.time_series.get_window(size)
    
    def get_numpy_window(
        self, 
        sensor_types: list[SensorType] | None = None,
        size: int | None = None
    ) -> dict[SensorType, any]:
        """
        numpy 배열로 변환된 윈도우 반환
        
        Args:
            sensor_types: 추출할 센서 타입 (None이면 모든 타입)
            size: 윈도우 크기 (None이면 기본값)
        
        Returns:
            센서 타입별 numpy 배열
        """
        with self._lock:
            return self.time_series.to_numpy(sensor_types, size)
    
    def clear(self) -> None:
        """버퍼 초기화"""
        with self._lock:
            self.time_series.clear()
    
    @property
    def is_running(self) -> bool:
        """연속 수집 중인지 여부"""
        return self._running
    
    @property
    def sample_count(self) -> int:
        """현재 버퍼의 샘플 수"""
        with self._lock:
            return self.time_series.count
    
    @property
    def has_enough_samples(self) -> bool:
        """윈도우 크기만큼 샘플이 있는지"""
        with self._lock:
            return self.time_series.has_enough_samples
    
    @property
    def registered_sensors(self) -> list[SensorType]:
        """등록된 센서 타입 목록"""
        return list(self.drivers.keys())
    
    def __enter__(self) -> "SensorHub":
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료 - 수집 중지"""
        self.stop_collection()
    
    def __repr__(self) -> str:
        return (
            f"SensorHub(sensors={len(self.drivers)}, "
            f"samples={self.sample_count}, "
            f"running={self._running})"
        )
