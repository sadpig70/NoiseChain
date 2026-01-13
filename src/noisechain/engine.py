"""
NoiseChain E2E 파이프라인

센서 데이터 수집부터 토큰 생성, 저장, 검증까지의 전체 워크플로우를 제공합니다.

MVP 핵심 컴포넌트:
1. SensorHub: 센서 데이터 수집
2. TokenBuilder: 토큰 생성
3. TokenSigner: 서명
4. TokenRepository: 저장
5. VerificationEngine: 검증
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

import numpy as np

from noisechain.crypto.keys import KeyManager, KeyPair
from noisechain.crypto.signing import TokenSigner, sign_token
from noisechain.pipeline.correlation import CorrelationSignature
from noisechain.pipeline.features import FeatureExtractor
from noisechain.sensors.hub import SensorHub
from noisechain.storage.repository import StorageConfig, TokenQuery, TokenRepository
from noisechain.token.builder import TokenBuilder
from noisechain.token.schema import PoXToken
from noisechain.verification.validator import (
    TokenValidator,
    VerificationConfig,
    VerificationEngine,
    VerificationReport,
)


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 노드 설정
    node_id: bytes = field(default_factory=lambda: uuid.uuid4().bytes)
    
    # 수집 설정
    sample_count: int = 256              # 수집 샘플 수
    sample_interval_ms: int = 10         # 샘플 간격 (ms)
    
    # 저장소 설정
    db_path: str = ":memory:"            # 데이터베이스 경로
    
    # 검증 설정
    max_risk_score: float = 80.0         # 최대 허용 위험 점수
    max_age_seconds: int = 86400 * 7     # 최대 토큰 수명
    
    # 키 설정
    keys_dir: Optional[str] = None       # 키 저장 디렉토리


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    token: Optional[PoXToken] = None
    verification: Optional[VerificationReport] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "success": self.success,
            "token_hash": self.token.compute_hash().hex() if self.token else None,
            "is_verified": self.verification.is_valid if self.verification else None,
            "error": self.error,
        }


class NoiseChainPipeline:
    """
    NoiseChain E2E 파이프라인
    
    센서 데이터 수집부터 검증까지의 전체 워크플로우를 관리합니다.
    
    Example:
        >>> pipeline = NoiseChainPipeline()
        >>> result = pipeline.generate_and_store()
        >>> print(result.success)
        True
    """
    
    def __init__(self, config: PipelineConfig | None = None):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config or PipelineConfig()
        
        # 컴포넌트 초기화
        self._sensor_hub = SensorHub()
        self._token_builder = TokenBuilder(node_id=self.config.node_id)
        self._feature_extractor = FeatureExtractor()
        self._correlation_sig = CorrelationSignature()
        
        # 키 관리
        self._key_manager = KeyManager(
            keys_dir=self.config.keys_dir
        )
        self._keypair = self._key_manager.get_or_create("default")
        
        # 저장소
        self._repository = TokenRepository(
            StorageConfig(db_path=self.config.db_path)
        )
        
        # 검증 엔진
        verification_config = VerificationConfig(
            max_risk_score=self.config.max_risk_score,
            max_age_seconds=self.config.max_age_seconds
        )
        self._verification_engine = VerificationEngine(config=verification_config)
        self._verification_engine.register_key(
            self.config.node_id,
            self._keypair.public_key
        )
    
    @property
    def node_id(self) -> bytes:
        """노드 ID"""
        return self.config.node_id
    
    @property
    def public_key(self) -> bytes:
        """공개 키"""
        return self._keypair.public_key
    
    @property
    def repository(self) -> TokenRepository:
        """토큰 저장소"""
        return self._repository
    
    @property
    def sensor_hub(self) -> SensorHub:
        """센서 허브"""
        return self._sensor_hub
    
    def collect_samples(self, count: int | None = None) -> dict[str, np.ndarray]:
        """
        센서 데이터 수집
        
        Args:
            count: 수집 샘플 수 (None이면 설정값)
        
        Returns:
            {sensor_name: samples} 딕셔너리
        """
        n = count or self.config.sample_count
        
        # 연속 수집
        self._sensor_hub.collect_samples(n)
        
        # NumPy 배열로 변환
        return self._sensor_hub.get_numpy_window()
    
    def generate_token(
        self,
        sensor_data: dict[str, np.ndarray] | None = None
    ) -> PoXToken:
        """
        토큰 생성
        
        Args:
            sensor_data: 센서 데이터 (None이면 새로 수집)
        
        Returns:
            생성된 PoXToken
        """
        # 데이터 수집
        if sensor_data is None:
            sensor_data = self.collect_samples()
        
        # 토큰 생성
        token = self._token_builder.from_sensor_data(sensor_data)
        
        # 서명
        signed_token = sign_token(token, self._keypair)
        
        return signed_token
    
    def generate_and_store(
        self,
        sensor_data: dict[str, np.ndarray] | None = None
    ) -> PipelineResult:
        """
        토큰 생성 및 저장
        
        Args:
            sensor_data: 센서 데이터 (None이면 새로 수집)
        
        Returns:
            PipelineResult
        """
        try:
            # 토큰 생성
            token = self.generate_token(sensor_data)
            
            # 저장
            self._repository.save(token)
            
            # 검증
            report = self._verification_engine.verify(token)
            
            return PipelineResult(
                success=True,
                token=token,
                verification=report
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                error=str(e)
            )
    
    def verify_token(self, token: PoXToken) -> VerificationReport:
        """
        토큰 검증
        
        Args:
            token: 검증할 토큰
        
        Returns:
            VerificationReport
        """
        return self._verification_engine.verify(token)
    
    def verify_by_hash(self, token_hash: bytes) -> Optional[VerificationReport]:
        """
        해시로 토큰 검증
        
        Args:
            token_hash: 토큰 해시
        
        Returns:
            VerificationReport (토큰 없으면 None)
        """
        token = self._repository.get_by_hash(token_hash)
        if token:
            return self.verify_token(token)
        return None
    
    def get_recent_tokens(self, limit: int = 10) -> list[PoXToken]:
        """최근 토큰 조회"""
        query = TokenQuery(node_id=self.config.node_id, limit=limit)
        return self._repository.query(query)
    
    def get_stats(self) -> dict:
        """파이프라인 통계"""
        storage_stats = self._repository.get_stats()
        
        return {
            "node_id": self.config.node_id.hex(),
            "public_key": self.public_key.hex()[:32] + "...",
            "total_tokens": storage_stats.total_tokens,
            "total_size_bytes": storage_stats.total_size_bytes,
            "unique_nodes": storage_stats.unique_nodes,
        }
    
    def close(self) -> None:
        """리소스 정리"""
        self._repository.close()
    
    def __enter__(self) -> "NoiseChainPipeline":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_pipeline(
    db_path: str | Path | None = None,
    node_id: bytes | None = None
) -> NoiseChainPipeline:
    """
    편의 함수: 파이프라인 생성
    
    Args:
        db_path: 데이터베이스 경로 (None이면 인메모리)
        node_id: 노드 ID (None이면 자동 생성)
    
    Returns:
        NoiseChainPipeline
    """
    config = PipelineConfig(
        db_path=str(db_path) if db_path else ":memory:",
        node_id=node_id or uuid.uuid4().bytes
    )
    return NoiseChainPipeline(config)
