"""
NoiseChain 검증 모듈 테스트

TokenValidator, VerificationEngine 단위 테스트입니다.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from noisechain.crypto.keys import KeyPair
from noisechain.crypto.signing import TokenSigner
from noisechain.token.schema import NoiseFingerprint, PoXToken
from noisechain.verification.validator import (
    TokenValidator,
    VerificationConfig,
    VerificationEngine,
    VerificationReport,
    VerificationStatus,
    VerificationStep,
    validate_token,
)


# ============================================================
# 테스트 헬퍼
# ============================================================

def create_test_token(
    node_id: bytes | None = None,
    timestamp_ns: int | None = None,
    risk_score: float = 10.0,
    signed: bool = False,
    keypair: KeyPair | None = None
) -> tuple[PoXToken, KeyPair | None]:
    """테스트용 토큰 생성"""
    fingerprint = NoiseFingerprint(
        feature_vector=bytes(64),
        correlation_hash=bytes(32),
        sensor_count=4,
        sample_count=256
    )
    token = PoXToken(
        node_id=node_id or uuid.uuid4().bytes,
        fingerprint=fingerprint,
        risk_score=risk_score,
        timestamp_ns=timestamp_ns or int(datetime.now().timestamp() * 1e9)
    )
    
    if signed:
        kp = keypair or KeyPair.generate()
        signer = TokenSigner(kp)
        return signer.sign(token).signed_token, kp
    
    return token, None


# ============================================================
# VerificationStep 테스트
# ============================================================

class TestVerificationStep:
    """VerificationStep 테스트"""
    
    def test_passed_status(self):
        """통과 상태"""
        step = VerificationStep(
            name="test",
            status=VerificationStatus.PASSED,
            message="OK"
        )
        
        assert step.passed is True
    
    def test_failed_status(self):
        """실패 상태"""
        step = VerificationStep(
            name="test",
            status=VerificationStatus.FAILED,
            message="Error"
        )
        
        assert step.passed is False
    
    def test_warning_counts_as_passed(self):
        """경고는 통과로 처리"""
        step = VerificationStep(
            name="test",
            status=VerificationStatus.WARNING,
            message="Warning"
        )
        
        assert step.passed is True


# ============================================================
# VerificationReport 테스트
# ============================================================

class TestVerificationReport:
    """VerificationReport 테스트"""
    
    def test_empty_report_is_valid(self):
        """빈 보고서는 유효"""
        report = VerificationReport(
            token_hash=bytes(32),
            timestamp=datetime.now()
        )
        
        assert report.is_valid is True
    
    def test_all_passed_is_valid(self):
        """모두 통과면 유효"""
        report = VerificationReport(
            token_hash=bytes(32),
            timestamp=datetime.now()
        )
        report.add_step(VerificationStep("s1", VerificationStatus.PASSED, ""))
        report.add_step(VerificationStep("s2", VerificationStatus.PASSED, ""))
        
        assert report.is_valid is True
        assert report.passed_count == 2
        assert report.failed_count == 0
    
    def test_one_failed_is_invalid(self):
        """하나라도 실패면 무효"""
        report = VerificationReport(
            token_hash=bytes(32),
            timestamp=datetime.now()
        )
        report.add_step(VerificationStep("s1", VerificationStatus.PASSED, ""))
        report.add_step(VerificationStep("s2", VerificationStatus.FAILED, ""))
        
        assert report.is_valid is False
        assert report.failed_count == 1
    
    def test_skipped_is_valid(self):
        """건너뜀은 유효"""
        report = VerificationReport(
            token_hash=bytes(32),
            timestamp=datetime.now()
        )
        report.add_step(VerificationStep("s1", VerificationStatus.SKIPPED, ""))
        
        assert report.is_valid is True
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        report = VerificationReport(
            token_hash=bytes(32),
            timestamp=datetime.now()
        )
        report.add_step(VerificationStep("schema", VerificationStatus.PASSED, "OK"))
        
        d = report.to_dict()
        
        assert "token_hash" in d
        assert "is_valid" in d
        assert len(d["steps"]) == 1


# ============================================================
# TokenValidator 테스트
# ============================================================

class TestTokenValidator:
    """TokenValidator 테스트"""
    
    def test_valid_unsigned_token(self):
        """서명 없는 유효한 토큰"""
        token, _ = create_test_token()
        config = VerificationConfig(verify_signature=False)
        validator = TokenValidator(config=config)
        
        report = validator.validate(token)
        
        assert report.is_valid is True
    
    def test_schema_validation_node_id(self):
        """스키마 검증: 노드 ID"""
        # 잘못된 노드 ID 길이는 PoXToken 생성 시 오류 발생
        # 정상 토큰으로 스키마 통과 확인
        token, _ = create_test_token()
        validator = TokenValidator()
        
        report = validator.validate(token)
        
        schema_step = next(s for s in report.steps if s.name == "schema")
        assert schema_step.status == VerificationStatus.PASSED
    
    def test_signature_verification_with_valid_key(self):
        """서명 검증: 올바른 키"""
        keypair = KeyPair.generate()
        token, _ = create_test_token(signed=True, keypair=keypair)
        
        validator = TokenValidator(public_key=keypair.public_key)
        report = validator.validate(token)
        
        sig_step = next(s for s in report.steps if s.name == "signature")
        assert sig_step.status == VerificationStatus.PASSED
    
    def test_signature_verification_with_wrong_key(self):
        """서명 검증: 잘못된 키"""
        keypair1 = KeyPair.generate()
        keypair2 = KeyPair.generate()
        token, _ = create_test_token(signed=True, keypair=keypair1)
        
        validator = TokenValidator(public_key=keypair2.public_key)
        report = validator.validate(token)
        
        sig_step = next(s for s in report.steps if s.name == "signature")
        assert sig_step.status == VerificationStatus.FAILED
    
    def test_signature_skipped_without_key(self):
        """서명 검증: 키 없으면 건너뜀"""
        token, _ = create_test_token(signed=True)
        
        validator = TokenValidator(public_key=None)
        report = validator.validate(token)
        
        sig_step = next(s for s in report.steps if s.name == "signature")
        assert sig_step.status == VerificationStatus.SKIPPED
    
    def test_timestamp_future_rejected(self):
        """타임스탬프: 미래 시간 거부"""
        future_ns = int((datetime.now() + timedelta(hours=1)).timestamp() * 1e9)
        token, _ = create_test_token(timestamp_ns=future_ns)
        
        config = VerificationConfig(verify_signature=False)
        validator = TokenValidator(config=config)
        report = validator.validate(token)
        
        ts_step = next(s for s in report.steps if s.name == "timestamp")
        assert ts_step.status == VerificationStatus.FAILED
    
    def test_timestamp_old_warning(self):
        """타임스탬프: 오래된 토큰 경고"""
        old_ns = int((datetime.now() - timedelta(days=30)).timestamp() * 1e9)
        token, _ = create_test_token(timestamp_ns=old_ns)
        
        config = VerificationConfig(verify_signature=False, max_age_seconds=86400)
        validator = TokenValidator(config=config)
        report = validator.validate(token)
        
        ts_step = next(s for s in report.steps if s.name == "timestamp")
        assert ts_step.status == VerificationStatus.WARNING
    
    def test_risk_score_high_warning(self):
        """위험 점수: 높으면 경고"""
        token, _ = create_test_token(risk_score=90.0)
        
        config = VerificationConfig(verify_signature=False, max_risk_score=80.0)
        validator = TokenValidator(config=config)
        report = validator.validate(token)
        
        risk_step = next(s for s in report.steps if s.name == "risk_score")
        assert risk_step.status == VerificationStatus.WARNING
    
    def test_quick_check(self):
        """빠른 검증"""
        token, _ = create_test_token()
        
        config = VerificationConfig(verify_signature=False)
        validator = TokenValidator(config=config)
        
        assert validator.quick_check(token) is True


# ============================================================
# VerificationEngine 테스트
# ============================================================

class TestVerificationEngine:
    """VerificationEngine 테스트"""
    
    def test_register_and_verify(self):
        """키 등록 및 검증"""
        keypair = KeyPair.generate()
        node_id = uuid.uuid4().bytes
        token, _ = create_test_token(node_id=node_id, signed=True, keypair=keypair)
        
        engine = VerificationEngine()
        engine.register_key(node_id, keypair.public_key)
        
        report = engine.verify(token)
        
        assert report.is_valid is True
    
    def test_verify_unregistered_node(self):
        """미등록 노드 검증 (서명 건너뜀)"""
        token, _ = create_test_token(signed=True)
        
        engine = VerificationEngine()
        report = engine.verify(token)
        
        # 서명 검증 건너뜀
        sig_step = next(s for s in report.steps if s.name == "signature")
        assert sig_step.status == VerificationStatus.SKIPPED
    
    def test_unregister_key(self):
        """키 제거"""
        engine = VerificationEngine()
        node_id = bytes(16)
        
        engine.register_key(node_id, bytes(32))
        assert engine.get_key(node_id) is not None
        
        result = engine.unregister_key(node_id)
        
        assert result is True
        assert engine.get_key(node_id) is None
    
    def test_verify_batch(self):
        """일괄 검증"""
        tokens = [create_test_token()[0] for _ in range(5)]
        
        config = VerificationConfig(verify_signature=False)
        engine = VerificationEngine(config=config)
        
        reports = engine.verify_batch(tokens)
        
        assert len(reports) == 5
        assert all(r.is_valid for r in reports)
    
    def test_registered_nodes(self):
        """등록된 노드 목록"""
        engine = VerificationEngine()
        
        for i in range(3):
            engine.register_key(bytes([i] * 16), bytes(32))
        
        assert len(engine.registered_nodes) == 3


# ============================================================
# 편의 함수 테스트
# ============================================================

class TestValidateTokenFunction:
    """validate_token 편의 함수 테스트"""
    
    def test_valid_token(self):
        """유효한 토큰"""
        token, _ = create_test_token()
        
        result = validate_token(token)
        
        assert result is True
    
    def test_with_public_key(self):
        """공개 키와 함께"""
        keypair = KeyPair.generate()
        token, _ = create_test_token(signed=True, keypair=keypair)
        
        result = validate_token(token, public_key=keypair.public_key)
        
        assert result is True


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationVerification:
    """검증 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우"""
        # 1. 노드 키 생성
        keypair = KeyPair.generate()
        node_id = uuid.uuid4().bytes
        
        # 2. 토큰 생성 및 서명
        token, _ = create_test_token(
            node_id=node_id,
            risk_score=15.0,
            signed=True,
            keypair=keypair
        )
        
        # 3. 엔진 설정
        config = VerificationConfig(
            max_risk_score=50.0,
            max_age_seconds=3600
        )
        engine = VerificationEngine(config=config)
        engine.register_key(node_id, keypair.public_key)
        
        # 4. 검증
        report = engine.verify(token)
        
        # 5. 결과 확인
        assert report.is_valid is True
        assert report.passed_count == 4  # schema, signature, timestamp, risk_score
        assert report.failed_count == 0
        
        # 6. 보고서 출력
        d = report.to_dict()
        assert d["is_valid"] is True
