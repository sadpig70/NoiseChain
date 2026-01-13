"""
NoiseChain NTP 클라이언트

NTP 서버와 동기화하여 정확한 시간 오프셋을 계산합니다.
시뮬레이션 환경에서는 로컬 시간을 사용합니다.
"""

import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class NTPResult:
    """NTP 동기화 결과"""
    offset_ns: int          # 로컬 시간과 NTP 시간의 차이 (나노초)
    delay_ns: int           # 왕복 지연 시간 (나노초)
    server: str             # 사용된 NTP 서버
    timestamp_ns: int       # NTP 시간 (나노초, Unix epoch 기준)
    success: bool           # 동기화 성공 여부
    error: Optional[str]    # 에러 메시지 (실패 시)


# NTP 상수
NTP_EPOCH = 2208988800  # 1900-01-01 ~ 1970-01-01 (초)
NTP_PORT = 123
NTP_PACKET_SIZE = 48


class NTPClient:
    """
    NTP 클라이언트
    
    NTP 서버와 동기화하여 로컬 시스템 시간과의 오프셋을 계산합니다.
    네트워크 불가 시 시뮬레이션 모드로 동작합니다.
    
    Attributes:
        servers: NTP 서버 목록
        timeout: 연결 타임아웃 (초)
        _last_offset_ns: 마지막 계산된 오프셋
    
    Example:
        >>> client = NTPClient()
        >>> result = client.sync()
        >>> if result.success:
        ...     print(f"오프셋: {result.offset_ns}ns")
    """
    
    # 기본 NTP 서버 풀
    DEFAULT_SERVERS = [
        "time.windows.com",
        "time.google.com",
        "pool.ntp.org",
        "time.cloudflare.com",
    ]
    
    def __init__(
        self, 
        servers: Optional[list[str]] = None,
        timeout: float = 2.0,
        simulation_mode: bool = False
    ):
        """
        Args:
            servers: NTP 서버 목록 (None이면 기본 서버 사용)
            timeout: 연결 타임아웃 (초)
            simulation_mode: 시뮬레이션 모드 강제 활성화
        """
        self.servers = servers or self.DEFAULT_SERVERS
        self.timeout = timeout
        self.simulation_mode = simulation_mode
        self._last_offset_ns: int = 0
        self._last_sync_time: Optional[float] = None
    
    def sync(self) -> NTPResult:
        """
        NTP 서버와 동기화
        
        서버 목록을 순회하며 첫 번째 성공하는 서버의 결과 반환.
        모든 서버 실패 시 시뮬레이션 결과 반환.
        
        Returns:
            NTPResult 객체
        """
        if self.simulation_mode:
            return self._simulate_sync()
        
        for server in self.servers:
            result = self._sync_with_server(server)
            if result.success:
                self._last_offset_ns = result.offset_ns
                self._last_sync_time = time.time()
                return result
        
        # 모든 서버 실패 시 시뮬레이션
        return self._simulate_sync()
    
    def _sync_with_server(self, server: str) -> NTPResult:
        """단일 NTP 서버와 동기화"""
        try:
            # NTP 패킷 생성 (클라이언트 모드, 버전 3)
            packet = bytearray(NTP_PACKET_SIZE)
            packet[0] = 0x1B  # LI=0, VN=3, Mode=3 (client)
            
            # 송신 시간 기록
            t1 = time.time_ns()
            
            # UDP 소켓 생성 및 전송
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.timeout)
            
            try:
                sock.sendto(bytes(packet), (server, NTP_PORT))
                response, _ = sock.recvfrom(NTP_PACKET_SIZE)
            finally:
                sock.close()
            
            # 수신 시간 기록
            t4 = time.time_ns()
            
            # NTP 응답 파싱 (전송 타임스탬프: 바이트 40-48)
            # 초 부분 (32비트) + 소수 부분 (32비트)
            ntp_seconds = struct.unpack("!I", response[40:44])[0]
            ntp_fraction = struct.unpack("!I", response[44:48])[0]
            
            # Unix 타임스탬프로 변환 (나노초)
            unix_seconds = ntp_seconds - NTP_EPOCH
            fraction_ns = int(ntp_fraction * 1e9 / (2**32))
            t3 = unix_seconds * 1_000_000_000 + fraction_ns
            
            # 오프셋 계산: offset = ((t2-t1) + (t3-t4)) / 2
            # 단순화: t2 ≈ t3 (서버 처리 시간 무시)
            offset = t3 - t4
            delay = t4 - t1
            
            return NTPResult(
                offset_ns=offset,
                delay_ns=delay,
                server=server,
                timestamp_ns=t3,
                success=True,
                error=None
            )
            
        except socket.timeout:
            return NTPResult(
                offset_ns=0, delay_ns=0, server=server,
                timestamp_ns=0, success=False, 
                error=f"타임아웃: {server}"
            )
        except socket.gaierror as e:
            return NTPResult(
                offset_ns=0, delay_ns=0, server=server,
                timestamp_ns=0, success=False,
                error=f"DNS 오류: {server} - {e}"
            )
        except Exception as e:
            return NTPResult(
                offset_ns=0, delay_ns=0, server=server,
                timestamp_ns=0, success=False,
                error=f"NTP 오류: {server} - {e}"
            )
    
    def _simulate_sync(self) -> NTPResult:
        """시뮬레이션 동기화 (오프셋 0)"""
        now = time.time_ns()
        # 시뮬레이션도 동기화 시간 기록
        self._last_sync_time = time.time()
        return NTPResult(
            offset_ns=0,
            delay_ns=0,
            server="simulation",
            timestamp_ns=now,
            success=True,
            error=None
        )
    
    @property
    def last_offset_ns(self) -> int:
        """마지막 동기화 오프셋 (나노초)"""
        return self._last_offset_ns
    
    @property
    def last_sync_age(self) -> Optional[float]:
        """마지막 동기화 이후 경과 시간 (초)"""
        if self._last_sync_time is None:
            return None
        return time.time() - self._last_sync_time
    
    def get_corrected_time_ns(self) -> int:
        """
        오프셋 보정된 현재 시간 (나노초)
        
        마지막 동기화 오프셋을 적용한 시간 반환.
        동기화한 적 없으면 로컬 시간 반환.
        """
        return time.time_ns() + self._last_offset_ns
