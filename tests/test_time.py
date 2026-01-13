"""
NoiseChain 시간 동기화 모듈 테스트

NTPClient, TimestampGenerator 단위 테스트입니다.
"""

import time

import pytest

from noisechain.time.ntp_client import NTPClient, NTPResult
from noisechain.time.timestamp import (
    TimestampConfig,
    TimestampGenerator,
    get_default_generator,
    now_ms,
    now_ns,
)


# ============================================================
# NTPClient 테스트
# ============================================================

class TestNTPResult:
    """NTPResult 테스트"""
    
    def test_success_result(self):
        """성공 결과 생성"""
        result = NTPResult(
            offset_ns=1000,
            delay_ns=500,
            server="test.server",
            timestamp_ns=1234567890_000000000,
            success=True,
            error=None
        )
        
        assert result.success is True
        assert result.offset_ns == 1000
        assert result.server == "test.server"
    
    def test_failure_result(self):
        """실패 결과 생성"""
        result = NTPResult(
            offset_ns=0,
            delay_ns=0,
            server="bad.server",
            timestamp_ns=0,
            success=False,
            error="타임아웃"
        )
        
        assert result.success is False
        assert result.error == "타임아웃"


class TestNTPClient:
    """NTPClient 테스트"""
    
    def test_default_servers(self):
        """기본 서버 목록 확인"""
        client = NTPClient()
        
        assert len(client.servers) > 0
        assert "time.windows.com" in client.servers or "time.google.com" in client.servers
    
    def test_custom_servers(self):
        """사용자 정의 서버"""
        servers = ["custom.time.server"]
        client = NTPClient(servers=servers)
        
        assert client.servers == servers
    
    def test_simulation_mode(self):
        """시뮬레이션 모드"""
        client = NTPClient(simulation_mode=True)
        
        result = client.sync()
        
        assert result.success is True
        assert result.server == "simulation"
        assert result.offset_ns == 0
    
    def test_get_corrected_time_ns(self):
        """보정된 시간 조회"""
        client = NTPClient(simulation_mode=True)
        client.sync()
        
        corrected = client.get_corrected_time_ns()
        local = time.time_ns()
        
        # 시뮬레이션 모드이므로 거의 동일해야 함
        assert abs(corrected - local) < 1_000_000_000  # 1초 이내
    
    def test_last_offset_ns(self):
        """마지막 오프셋 확인"""
        client = NTPClient(simulation_mode=True)
        
        # 동기화 전
        assert client.last_offset_ns == 0
        
        # 동기화 후
        client.sync()
        assert client.last_offset_ns == 0  # 시뮬레이션은 항상 0
    
    def test_last_sync_age(self):
        """마지막 동기화 경과 시간"""
        client = NTPClient(simulation_mode=True)
        
        # 동기화 전
        assert client.last_sync_age is None
        
        # 동기화 후
        client.sync()
        time.sleep(0.01)
        
        age = client.last_sync_age
        assert age is not None
        assert age >= 0.01


# ============================================================
# TimestampGenerator 테스트
# ============================================================

class TestTimestampConfig:
    """TimestampConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        config = TimestampConfig()
        
        assert config.ntp_enabled is True
        assert config.sync_interval == 3600.0
        assert config.monotonic_check is True


class TestTimestampGenerator:
    """TimestampGenerator 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        gen = TimestampGenerator()
        
        assert gen.config is not None
        assert gen.ntp_client is not None
    
    def test_custom_config(self):
        """사용자 정의 설정"""
        config = TimestampConfig(
            ntp_enabled=False,
            sync_interval=60.0
        )
        gen = TimestampGenerator(config=config)
        
        assert gen.config.ntp_enabled is False
        assert gen.config.sync_interval == 60.0
    
    def test_sync(self):
        """NTP 동기화"""
        config = TimestampConfig(ntp_enabled=False)  # 시뮬레이션 모드
        gen = TimestampGenerator(config=config)
        
        result = gen.sync()
        
        assert result.success is True
        assert gen.is_synced is True
    
    def test_now_ns(self):
        """나노초 타임스탬프 생성"""
        gen = TimestampGenerator()
        
        ts = gen.now_ns()
        
        assert ts > 0
        assert ts < time.time_ns() + 1_000_000_000  # 합리적 범위
    
    def test_now_ms(self):
        """밀리초 타임스탬프 생성"""
        gen = TimestampGenerator()
        
        ts = gen.now_ms()
        
        assert ts > 0
        assert ts == gen.now_ns() // 1_000_000 or abs(ts - gen.now_ns() // 1_000_000) <= 1
    
    def test_now_seconds(self):
        """초 타임스탬프 생성"""
        gen = TimestampGenerator()
        
        ts = gen.now_seconds()
        
        assert ts > 0
        assert abs(ts - time.time()) < 1  # 1초 이내 차이
    
    def test_monotonic_check(self):
        """단조 증가 보장"""
        config = TimestampConfig(monotonic_check=True)
        gen = TimestampGenerator(config=config)
        
        timestamps = [gen.now_ns() for _ in range(100)]
        
        # 모든 타임스탬프가 증가해야 함
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]
    
    def test_sync_offset(self):
        """동기화 오프셋 확인"""
        config = TimestampConfig(ntp_enabled=False)
        gen = TimestampGenerator(config=config)
        gen.sync()
        
        # 시뮬레이션 모드는 오프셋 0
        assert gen.sync_offset_ns == 0
        assert gen.sync_offset_ms == 0.0
    
    def test_needs_resync(self):
        """재동기화 필요 여부"""
        config = TimestampConfig(
            ntp_enabled=False,
            sync_interval=0.01  # 10ms
        )
        gen = TimestampGenerator(config=config)
        
        # 동기화 전
        assert gen.needs_resync() is True
        
        # 동기화 직후
        gen.sync()
        assert gen.needs_resync() is False
        
        # 시간 경과 후
        time.sleep(0.02)
        assert gen.needs_resync() is True
    
    def test_ensure_synced(self):
        """동기화 보장"""
        config = TimestampConfig(ntp_enabled=False)
        gen = TimestampGenerator(config=config)
        
        result = gen.ensure_synced()
        
        assert result.success is True
        assert gen.is_synced is True
    
    def test_create_time_window(self):
        """시간창 생성"""
        gen = TimestampGenerator()
        
        start_ns, end_ns = gen.create_time_window(1000)  # 1초
        
        assert start_ns < end_ns
        assert end_ns - start_ns == 1000 * 1_000_000  # 1000ms = 1e9 ns
    
    def test_repr(self):
        """문자열 표현"""
        config = TimestampConfig(ntp_enabled=False)
        gen = TimestampGenerator(config=config)
        
        # 동기화 전
        assert "not synced" in repr(gen)
        
        # 동기화 후
        gen.sync()
        assert "synced" in repr(gen)


class TestConvenienceFunctions:
    """편의 함수 테스트"""
    
    def test_get_default_generator(self):
        """기본 생성기 싱글톤"""
        gen1 = get_default_generator()
        gen2 = get_default_generator()
        
        assert gen1 is gen2
    
    def test_now_ns_function(self):
        """now_ns 편의 함수"""
        ts = now_ns()
        
        assert ts > 0
        assert isinstance(ts, int)
    
    def test_now_ms_function(self):
        """now_ms 편의 함수"""
        ts = now_ms()
        
        assert ts > 0
        assert isinstance(ts, int)


class TestIntegrationTime:
    """시간 모듈 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 생성 → 동기화 → 타임스탬프 → 시간창"""
        # 시뮬레이션 설정
        config = TimestampConfig(
            ntp_enabled=False,
            monotonic_check=True
        )
        
        gen = TimestampGenerator(config=config)
        
        # 동기화
        result = gen.sync()
        assert result.success is True
        
        # 연속 타임스탬프 생성
        timestamps = []
        for _ in range(50):
            timestamps.append(gen.now_ns())
        
        # 단조 증가 확인
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]
        
        # 시간창 생성
        start, end = gen.create_time_window(100)  # 100ms
        assert end - start == 100_000_000  # 100ms in ns
