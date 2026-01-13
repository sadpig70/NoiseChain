"""
NoiseChain 토큰 모듈 테스트

PoXToken, NoiseFingerprint, TokenBuilder 단위 테스트입니다.
"""

import uuid

import numpy as np
import pytest

from noisechain.token.builder import (
    RiskConfig,
    RiskScorer,
    TokenBuilder,
    build_token,
)
from noisechain.token.schema import (
    NoiseFingerprint,
    PoXToken,
    TokenType,
    TokenVersion,
)


# ============================================================
# NoiseFingerprint 테스트
# ============================================================

class TestNoiseFingerprint:
    """NoiseFingerprint 테스트"""
    
    def test_valid_initialization(self):
        """유효한 초기화"""
        fp = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
        
        assert len(fp.feature_vector) == 64
        assert len(fp.correlation_hash) == 32
        assert fp.sensor_count == 4
        assert fp.sample_count == 256
    
    def test_invalid_hash_length(self):
        """잘못된 해시 길이"""
        with pytest.raises(ValueError):
            NoiseFingerprint(
                feature_vector=bytes(64),
                correlation_hash=bytes(16),  # 32 필요
                sensor_count=4,
                sample_count=256
            )
    
    def test_invalid_sensor_count(self):
        """잘못된 센서 개수"""
        with pytest.raises(ValueError):
            NoiseFingerprint(
                feature_vector=bytes(64),
                correlation_hash=bytes(32),
                sensor_count=0,  # 최소 1
                sample_count=256
            )
    
    def test_to_bytes(self):
        """바이트 직렬화"""
        fp = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
        
        data = fp.to_bytes()
        
        assert len(data) == 99
    
    def test_from_bytes_roundtrip(self):
        """직렬화/역직렬화 왕복"""
        original = NoiseFingerprint(
            feature_vector=bytes(range(64)),
            correlation_hash=bytes(range(32)),
            sensor_count=4,
            sample_count=1000
        )
        
        data = original.to_bytes()
        restored = NoiseFingerprint.from_bytes(data)
        
        assert restored.feature_vector == original.feature_vector
        assert restored.correlation_hash == original.correlation_hash
        assert restored.sensor_count == original.sensor_count
        assert restored.sample_count == original.sample_count
    
    def test_size_property(self):
        """크기 속성"""
        fp = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=1,
            sample_count=1
        )
        
        assert fp.size == 99


# ============================================================
# PoXToken 테스트
# ============================================================

class TestPoXToken:
    """PoXToken 테스트"""
    
    @pytest.fixture
    def fingerprint(self):
        """테스트용 지문"""
        return NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
    
    def test_valid_initialization(self, fingerprint):
        """유효한 초기화"""
        node_id = uuid.uuid4().bytes
        
        token = PoXToken(
            node_id=node_id,
            fingerprint=fingerprint,
            risk_score=5.5
        )
        
        assert token.node_id == node_id
        assert token.risk_score == 5.5
        assert token.version == TokenVersion.V1_MVP
        assert token.is_signed is False
    
    def test_invalid_node_id_length(self, fingerprint):
        """잘못된 노드 ID 길이"""
        with pytest.raises(ValueError):
            PoXToken(
                node_id=bytes(8),  # 16 필요
                fingerprint=fingerprint,
            )
    
    def test_invalid_risk_score_range(self, fingerprint):
        """잘못된 위험 점수 범위"""
        with pytest.raises(ValueError):
            PoXToken(
                node_id=bytes(16),
                fingerprint=fingerprint,
                risk_score=150.0  # 0-100 범위
            )
    
    def test_to_bytes_unsigned(self, fingerprint):
        """서명 없는 직렬화"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
            risk_score=10.0
        )
        
        data = token.to_bytes()
        
        assert len(data) == 135  # 서명 제외
    
    def test_to_bytes_signed(self, fingerprint):
        """서명 있는 직렬화"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
            risk_score=10.0,
            signature=bytes(64)
        )
        
        data = token.to_bytes()
        
        assert len(data) == 199  # 서명 포함
    
    def test_from_bytes_roundtrip(self, fingerprint):
        """직렬화/역직렬화 왕복"""
        node_id = uuid.uuid4().bytes
        original = PoXToken(
            node_id=node_id,
            fingerprint=fingerprint,
            risk_score=25.5,
            signature=bytes(64)
        )
        
        data = original.to_bytes()
        restored = PoXToken.from_bytes(data)
        
        assert restored.node_id == original.node_id
        assert restored.risk_score == original.risk_score
        assert restored.signature == original.signature
    
    def test_compute_hash(self, fingerprint):
        """해시 계산"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
        )
        
        hash1 = token.compute_hash()
        hash2 = token.compute_hash()
        
        assert len(hash1) == 32
        assert hash1 == hash2  # 결정적
    
    def test_to_dict(self, fingerprint):
        """딕셔너리 변환"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
            risk_score=5.0
        )
        
        d = token.to_dict()
        
        assert "version" in d
        assert "node_id" in d
        assert "risk_score" in d
        assert d["risk_score"] == 5.0
    
    def test_to_json(self, fingerprint):
        """JSON 변환"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
        )
        
        json_str = token.to_json()
        
        assert "version" in json_str
        assert "node_id" in json_str
    
    def test_timestamp_datetime(self, fingerprint):
        """타임스탬프 datetime 변환"""
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint,
        )
        
        dt = token.timestamp_datetime
        
        assert dt.year >= 2024


# ============================================================
# RiskScorer 테스트
# ============================================================

class TestRiskScorer:
    """RiskScorer 테스트"""
    
    def test_default_config(self):
        """기본 설정"""
        scorer = RiskScorer()
        
        assert scorer.config.entropy_weight > 0
        assert scorer.config.variance_weight > 0
    
    def test_compute_empty_features(self):
        """빈 특징으로 계산"""
        from noisechain.pipeline.correlation import CorrelationResult
        
        scorer = RiskScorer()
        correlation = CorrelationResult(
            feature_vector=np.zeros(16),
            correlation_hash=bytes(32),
            sensor_count=0
        )
        
        score = scorer.compute({}, correlation)
        
        assert 0 <= score <= 100


# ============================================================
# TokenBuilder 테스트
# ============================================================

class TestTokenBuilder:
    """TokenBuilder 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        builder = TokenBuilder()
        
        assert len(builder.node_id) == 16
        assert builder.version == TokenVersion.V1_MVP
    
    def test_custom_node_id(self):
        """사용자 정의 노드 ID"""
        node_id = uuid.uuid4().bytes
        builder = TokenBuilder(node_id=node_id)
        
        assert builder.node_id == node_id
    
    def test_from_sensor_data(self):
        """센서 데이터에서 토큰 생성"""
        np.random.seed(42)
        sensor_data = {
            "sensor1": np.sin(np.linspace(0, 4*np.pi, 100)),
            "sensor2": np.random.randn(100)
        }
        
        builder = TokenBuilder()
        token = builder.from_sensor_data(sensor_data)
        
        assert token is not None
        assert token.fingerprint.sensor_count == 2
        assert 0 <= token.risk_score <= 100
    
    def test_from_sensor_data_with_timestamp(self):
        """커스텀 타임스탬프로 토큰 생성"""
        sensor_data = {"sensor1": np.array([1.0, 2.0, 3.0])}
        timestamp = 1700000000_000000000  # 지정 시간
        
        builder = TokenBuilder()
        token = builder.from_sensor_data(sensor_data, timestamp_ns=timestamp)
        
        assert token.timestamp_ns == timestamp


class TestBuildTokenFunction:
    """build_token 편의 함수 테스트"""
    
    def test_basic_usage(self):
        """기본 사용법"""
        sensor_data = {
            "sensor1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        token = build_token(sensor_data)
        
        assert token is not None
        assert token.is_signed is False


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationToken:
    """토큰 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 센서 → 빌더 → 토큰 → 직렬화 → 복원"""
        # 센서 데이터 시뮬레이션
        np.random.seed(42)
        sensor_data = {
            "cpu_temp": 50 + 10 * np.sin(np.linspace(0, 2*np.pi, 256)),
            "os_entropy": np.random.randint(0, 256, 256).astype(float),
            "clock_jitter": np.random.randn(256) * 100,
            "synthetic": np.random.randn(256) * 0.5
        }
        
        # 토큰 생성
        node_id = uuid.uuid4().bytes
        builder = TokenBuilder(node_id=node_id)
        token = builder.from_sensor_data(sensor_data)
        
        # 검증
        assert token.node_id == node_id
        assert token.fingerprint.sensor_count == 4
        assert 0 <= token.risk_score <= 100
        
        # 직렬화
        data = token.to_bytes()
        assert len(data) == 135  # 서명 미포함
        
        # 복원
        restored = PoXToken.from_bytes(data)
        assert restored.node_id == token.node_id
        # risk_score는 0.01 단위로 양자화되므로 근사 비교
        assert abs(restored.risk_score - token.risk_score) < 0.02
        
        # JSON
        json_str = token.to_json()
        assert "risk_score" in json_str
