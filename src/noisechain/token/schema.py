"""
NoiseChain PoXToken 스키마

Proof-of-eXistence Token의 데이터 구조를 정의합니다.
MVP 설계서에 따라 ~200 바이트 크기의 경량 토큰을 구현합니다.

설계 기반: NoiseChain_MVP_Design.md 섹션 4 (PoXToken 스키마)
"""

import hashlib
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional


class TokenVersion(IntEnum):
    """토큰 버전"""
    V1_MVP = 1      # MVP 버전
    V2_FUTURE = 2   # 향후 확장


class TokenType(IntEnum):
    """토큰 타입"""
    STANDARD = 0          # 표준 증명 토큰
    LIGHTWEIGHT = 1       # 경량 토큰 (해시만)
    EXTENDED = 2          # 확장 토큰 (메타데이터 포함)


@dataclass
class NoiseFingerprint:
    """
    노이즈 지문
    
    센서 데이터에서 추출된 고유 특징 벡터와 상관 해시.
    
    Attributes:
        feature_vector: 정규화된 특징 벡터 (16 floats, 64 bytes)
        correlation_hash: 상관 서명 해시 (32 bytes)
        sensor_count: 사용된 센서 수 (1 byte)
        sample_count: 사용된 샘플 수 (2 bytes)
    """
    feature_vector: bytes  # 64 bytes (16 x float32)
    correlation_hash: bytes  # 32 bytes (SHA3-256)
    sensor_count: int
    sample_count: int
    
    def __post_init__(self):
        """데이터 검증"""
        if len(self.correlation_hash) != 32:
            raise ValueError(f"correlation_hash는 32 bytes여야 합니다: {len(self.correlation_hash)}")
        if self.sensor_count < 1 or self.sensor_count > 255:
            raise ValueError(f"sensor_count 범위 초과: {self.sensor_count}")
        if self.sample_count < 1 or self.sample_count > 65535:
            raise ValueError(f"sample_count 범위 초과: {self.sample_count}")
    
    def to_bytes(self) -> bytes:
        """바이트 직렬화 (최대 99 bytes)"""
        return (
            self.feature_vector +         # 64 bytes
            self.correlation_hash +       # 32 bytes
            struct.pack('!B', self.sensor_count) +   # 1 byte
            struct.pack('!H', self.sample_count)     # 2 bytes
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "NoiseFingerprint":
        """바이트 역직렬화"""
        if len(data) < 99:
            raise ValueError(f"데이터 길이 부족: {len(data)}")
        
        feature_vector = data[:64]
        correlation_hash = data[64:96]
        sensor_count = struct.unpack('!B', data[96:97])[0]
        sample_count = struct.unpack('!H', data[97:99])[0]
        
        return cls(
            feature_vector=feature_vector,
            correlation_hash=correlation_hash,
            sensor_count=sensor_count,
            sample_count=sample_count
        )
    
    @property
    def size(self) -> int:
        """바이트 크기"""
        return 99  # 고정 크기


@dataclass
class PoXToken:
    """
    Proof-of-eXistence Token
    
    노이즈 기반 존재 증명 토큰의 핵심 데이터 구조.
    
    Layout (~200 bytes):
        - version: 1 byte
        - type: 1 byte
        - timestamp_ns: 8 bytes (나노초)
        - node_id: 16 bytes (UUID)
        - fingerprint: 99 bytes
        - risk_score: 2 bytes (0-10000, 0.01 단위)
        - signature: 64 bytes (Ed25519)
        - nonce: 8 bytes (리플레이 방지)
    
    총 크기: 199 bytes
    
    Attributes:
        version: 토큰 버전
        token_type: 토큰 타입
        timestamp_ns: 생성 타임스탬프 (나노초)
        node_id: 노드 식별자 (16 bytes)
        fingerprint: 노이즈 지문
        risk_score: 위험 점수 (0.0 ~ 100.0)
        signature: Ed25519 서명 (64 bytes, 서명 전 None)
        nonce: 논스 (리플레이 방지)
    
    Example:
        >>> token = PoXToken(
        ...     node_id=uuid.uuid4().bytes,
        ...     fingerprint=fingerprint,
        ...     risk_score=5.5
        ... )
        >>> data = token.to_bytes()
    """
    node_id: bytes
    fingerprint: NoiseFingerprint
    risk_score: float = 0.0
    version: TokenVersion = TokenVersion.V1_MVP
    token_type: TokenType = TokenType.STANDARD
    timestamp_ns: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1e9))
    signature: Optional[bytes] = None
    nonce: int = field(default_factory=lambda: int.from_bytes(__import__('os').urandom(8), 'big'))
    
    # 메타데이터 (직렬화 제외)
    metadata: dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """데이터 검증"""
        if len(self.node_id) != 16:
            raise ValueError(f"node_id는 16 bytes여야 합니다: {len(self.node_id)}")
        if self.risk_score < 0.0 or self.risk_score > 100.0:
            raise ValueError(f"risk_score 범위 초과: {self.risk_score}")
        if self.signature is not None and len(self.signature) != 64:
            raise ValueError(f"signature는 64 bytes여야 합니다: {len(self.signature)}")
    
    def to_bytes(self, include_signature: bool = True) -> bytes:
        """
        바이트 직렬화
        
        Args:
            include_signature: 서명 포함 여부 (서명 생성 시 False)
        
        Returns:
            직렬화된 바이트
        """
        # risk_score를 0-10000 정수로 변환 (0.01 단위)
        risk_int = int(self.risk_score * 100)
        
        data = (
            struct.pack('!B', self.version) +       # 1 byte
            struct.pack('!B', self.token_type) +    # 1 byte
            struct.pack('!Q', self.timestamp_ns) +  # 8 bytes
            self.node_id +                          # 16 bytes
            self.fingerprint.to_bytes() +           # 99 bytes
            struct.pack('!H', risk_int) +           # 2 bytes
            struct.pack('!Q', self.nonce)           # 8 bytes
        )
        
        if include_signature and self.signature is not None:
            data += self.signature  # 64 bytes
        
        return data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "PoXToken":
        """바이트 역직렬화"""
        if len(data) < 135:  # 최소: 서명 제외
            raise ValueError(f"데이터 길이 부족: {len(data)}")
        
        offset = 0
        
        version = TokenVersion(struct.unpack('!B', data[offset:offset+1])[0])
        offset += 1
        
        token_type = TokenType(struct.unpack('!B', data[offset:offset+1])[0])
        offset += 1
        
        timestamp_ns = struct.unpack('!Q', data[offset:offset+8])[0]
        offset += 8
        
        node_id = data[offset:offset+16]
        offset += 16
        
        fingerprint = NoiseFingerprint.from_bytes(data[offset:offset+99])
        offset += 99
        
        risk_int = struct.unpack('!H', data[offset:offset+2])[0]
        risk_score = risk_int / 100.0
        offset += 2
        
        nonce = struct.unpack('!Q', data[offset:offset+8])[0]
        offset += 8
        
        signature = None
        if len(data) >= offset + 64:
            signature = data[offset:offset+64]
        
        return cls(
            version=version,
            token_type=token_type,
            timestamp_ns=timestamp_ns,
            node_id=node_id,
            fingerprint=fingerprint,
            risk_score=risk_score,
            signature=signature,
            nonce=nonce
        )
    
    def get_signing_data(self) -> bytes:
        """서명 대상 데이터 (서명 제외)"""
        return self.to_bytes(include_signature=False)
    
    def compute_hash(self) -> bytes:
        """토큰 해시 계산 (SHA3-256)"""
        return hashlib.sha3_256(self.to_bytes()).digest()
    
    @property
    def is_signed(self) -> bool:
        """서명 여부"""
        return self.signature is not None
    
    @property
    def size(self) -> int:
        """바이트 크기"""
        base = 135  # 서명 제외
        return base + 64 if self.is_signed else base
    
    @property
    def timestamp_datetime(self) -> datetime:
        """타임스탬프를 datetime으로 변환"""
        return datetime.fromtimestamp(self.timestamp_ns / 1e9)
    
    @property
    def node_id_hex(self) -> str:
        """노드 ID 16진수 문자열"""
        return self.node_id.hex()
    
    def to_dict(self) -> dict:
        """딕셔너리 변환 (JSON 직렬화용)"""
        return {
            "version": self.version,
            "type": self.token_type,
            "timestamp_ns": self.timestamp_ns,
            "timestamp": self.timestamp_datetime.isoformat(),
            "node_id": self.node_id_hex,
            "fingerprint": {
                "correlation_hash": self.fingerprint.correlation_hash.hex(),
                "sensor_count": self.fingerprint.sensor_count,
                "sample_count": self.fingerprint.sample_count,
            },
            "risk_score": self.risk_score,
            "signature": self.signature.hex() if self.signature else None,
            "nonce": self.nonce,
            "size_bytes": self.size,
        }
    
    def to_json(self) -> str:
        """JSON 문자열 변환"""
        return json.dumps(self.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        return (
            f"PoXToken(v{self.version}, "
            f"node={self.node_id_hex[:8]}..., "
            f"risk={self.risk_score:.2f}, "
            f"signed={self.is_signed})"
        )
