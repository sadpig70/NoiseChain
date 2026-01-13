"""
NoiseChain 타임스탬프 생성기

나노초 정밀도의 타임스탬프를 생성하고 NTP 오프셋을 적용합니다.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from noisechain.time.ntp_client import NTPClient, NTPResult


@dataclass
class TimestampConfig:
    """타임스탬프 생성기 설정"""
    ntp_enabled: bool = True          # NTP 동기화 활성화
    ntp_servers: list[str] | None = None  # NTP 서버 목록
    ntp_timeout: float = 2.0          # NTP 타임아웃
    sync_interval: float = 3600.0     # 자동 동기화 간격 (초)
    monotonic_check: bool = True      # 단조 증가 검증


class TimestampGenerator:
    """
    타임스탬프 생성기
    
    나노초 정밀도의 타임스탬프를 생성하며, NTP 동기화를 통해
    여러 장비 간 시간 일관성을 보장합니다.
    
    Attributes:
        config: 생성기 설정
        ntp_client: NTP 클라이언트 인스턴스
    
    Example:
        >>> gen = TimestampGenerator()
        >>> gen.sync()  # NTP 동기화
        >>> ts = gen.now_ns()  # 현재 시간 (나노초)
        >>> ts_ms = gen.now_ms()  # 현재 시간 (밀리초)
    """
    
    def __init__(self, config: TimestampConfig | None = None):
        """
        Args:
            config: 생성기 설정 (None이면 기본값 사용)
        """
        self.config = config or TimestampConfig()
        
        # NTP 클라이언트 초기화
        self.ntp_client = NTPClient(
            servers=self.config.ntp_servers,
            timeout=self.config.ntp_timeout,
            simulation_mode=not self.config.ntp_enabled
        )
        
        # 상태
        self._last_timestamp_ns: int = 0
        self._lock = threading.Lock()
        self._sync_result: Optional[NTPResult] = None
    
    def sync(self) -> NTPResult:
        """
        NTP 동기화 수행
        
        Returns:
            NTPResult 객체 (동기화 결과)
        """
        result = self.ntp_client.sync()
        self._sync_result = result
        return result
    
    def now_ns(self) -> int:
        """
        현재 시간 (나노초, Unix epoch 기준)
        
        NTP 오프셋을 적용한 보정된 시간을 반환합니다.
        monotonic_check 활성화 시 단조 증가를 보장합니다.
        
        Returns:
            나노초 타임스탬프
        """
        # 기본 시간: NTP 보정 적용
        timestamp = self.ntp_client.get_corrected_time_ns()
        
        # 단조 증가 보장
        if self.config.monotonic_check:
            with self._lock:
                if timestamp <= self._last_timestamp_ns:
                    timestamp = self._last_timestamp_ns + 1
                self._last_timestamp_ns = timestamp
        
        return timestamp
    
    def now_ms(self) -> int:
        """
        현재 시간 (밀리초, Unix epoch 기준)
        
        Returns:
            밀리초 타임스탬프
        """
        return self.now_ns() // 1_000_000
    
    def now_seconds(self) -> float:
        """
        현재 시간 (초, Unix epoch 기준)
        
        Returns:
            초 타임스탬프 (소수점 포함)
        """
        return self.now_ns() / 1_000_000_000
    
    @property
    def is_synced(self) -> bool:
        """NTP 동기화 완료 여부"""
        return self._sync_result is not None and self._sync_result.success
    
    @property
    def sync_offset_ns(self) -> int:
        """현재 NTP 오프셋 (나노초)"""
        return self.ntp_client.last_offset_ns
    
    @property
    def sync_offset_ms(self) -> float:
        """현재 NTP 오프셋 (밀리초)"""
        return self.ntp_client.last_offset_ns / 1_000_000
    
    @property
    def sync_age(self) -> Optional[float]:
        """마지막 동기화 이후 경과 시간 (초)"""
        return self.ntp_client.last_sync_age
    
    def needs_resync(self) -> bool:
        """
        재동기화 필요 여부
        
        sync_interval 이후 경과했으면 True
        """
        age = self.sync_age
        if age is None:
            return True
        return age > self.config.sync_interval
    
    def ensure_synced(self) -> NTPResult:
        """
        동기화 보장
        
        필요 시 재동기화 수행.
        """
        if self.needs_resync():
            return self.sync()
        return self._sync_result or self.sync()
    
    def create_time_window(self, duration_ms: int) -> tuple[int, int]:
        """
        시간창 생성
        
        현재 시간을 기준으로 지정된 지속 시간의 윈도우 생성.
        
        Args:
            duration_ms: 지속 시간 (밀리초)
        
        Returns:
            (start_ns, end_ns) 튜플
        """
        end_ns = self.now_ns()
        start_ns = end_ns - (duration_ms * 1_000_000)
        return start_ns, end_ns
    
    def __repr__(self) -> str:
        synced = "synced" if self.is_synced else "not synced"
        offset = f"offset={self.sync_offset_ms:.2f}ms" if self.is_synced else ""
        return f"TimestampGenerator({synced}, {offset})"


# 전역 기본 생성기 (편의용)
_default_generator: Optional[TimestampGenerator] = None


def get_default_generator() -> TimestampGenerator:
    """기본 타임스탬프 생성기 반환 (싱글톤)"""
    global _default_generator
    if _default_generator is None:
        _default_generator = TimestampGenerator()
    return _default_generator


def now_ns() -> int:
    """편의 함수: 현재 시간 (나노초)"""
    return get_default_generator().now_ns()


def now_ms() -> int:
    """편의 함수: 현재 시간 (밀리초)"""
    return get_default_generator().now_ms()
