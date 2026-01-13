"""
NoiseChain 토큰 검증 엔진

PoXToken의 유효성을 다단계로 검증합니다.
스키마, 서명, 상관관계, 위험점수를 종합적으로 평가합니다.

설계 기반: NoiseChain_MVP_Design.md 섹션 7 (검증 체크리스트)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from noisechain.crypto.keys import KeyPair
from noisechain.crypto.signing import TokenVerifier, VerificationResult
from noisechain.pipeline.correlation import CorrelationResult, CorrelationSignature
from noisechain.token.schema import NoiseFingerprint, PoXToken


class VerificationStatus(Enum):
    """검증 상태"""
    PASSED = auto()       # 검증 통과
    FAILED = auto()       # 검증 실패
    SKIPPED = auto()      # 건너뜀 (선택 검증)
    WARNING = auto()      # 경고 (통과하지만 주의)


@dataclass
class VerificationStep:
    """단일 검증 단계 결과"""
    name: str
    status: VerificationStatus
    message: str = ""
    details: dict = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        return self.status in (VerificationStatus.PASSED, VerificationStatus.WARNING)


@dataclass
class VerificationReport:
    """검증 보고서"""
    token_hash: bytes
    timestamp: datetime
    steps: list[VerificationStep] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """전체 유효성 (필수 단계 모두 통과)"""
        return all(
            step.passed or step.status == VerificationStatus.SKIPPED
            for step in self.steps
        )
    
    @property
    def passed_count(self) -> int:
        """통과 단계 수"""
        return sum(1 for s in self.steps if s.passed)
    
    @property
    def failed_count(self) -> int:
        """실패 단계 수"""
        return sum(1 for s in self.steps if s.status == VerificationStatus.FAILED)
    
    def add_step(self, step: VerificationStep) -> None:
        """단계 추가"""
        self.steps.append(step)
    
    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "token_hash": self.token_hash.hex(),
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.name,
                    "message": s.message,
                }
                for s in self.steps
            ]
        }


@dataclass
class VerificationConfig:
    """검증 설정"""
    verify_schema: bool = True           # 스키마 검증
    verify_signature: bool = True        # 서명 검증
    verify_timestamp: bool = True        # 타임스탬프 검증
    verify_risk_threshold: bool = True   # 위험 점수 임계값
    max_age_seconds: int = 86400 * 7     # 최대 토큰 수명 (7일)
    max_risk_score: float = 80.0         # 최대 허용 위험 점수
    min_sensor_count: int = 1            # 최소 센서 수


class TokenValidator:
    """
    토큰 검증기
    
    PoXToken의 유효성을 다단계로 검증합니다.
    
    검증 단계:
    1. 스키마 검증: 필수 필드, 크기 제약
    2. 서명 검증: Ed25519 서명 유효성
    3. 타임스탬프 검증: 시간 범위, 미래 시간 거부
    4. 위험 점수 검증: 임계값 이하
    
    Example:
        >>> validator = TokenValidator(public_key=keypair.public_key)
        >>> report = validator.validate(token)
        >>> print(report.is_valid)
    """
    
    def __init__(
        self,
        public_key: bytes | None = None,
        config: VerificationConfig | None = None
    ):
        """
        Args:
            public_key: 서명 검증용 공개 키 (None이면 서명 검증 건너뜀)
            config: 검증 설정
        """
        self.public_key = public_key
        self.config = config or VerificationConfig()
        self._signature_verifier = TokenVerifier()
    
    def validate(self, token: PoXToken) -> VerificationReport:
        """
        토큰 검증
        
        Args:
            token: 검증할 PoXToken
        
        Returns:
            VerificationReport
        """
        report = VerificationReport(
            token_hash=token.compute_hash(),
            timestamp=datetime.now()
        )
        
        # 1. 스키마 검증
        if self.config.verify_schema:
            report.add_step(self._verify_schema(token))
        
        # 2. 서명 검증
        if self.config.verify_signature and self.public_key:
            report.add_step(self._verify_signature(token))
        elif self.config.verify_signature and not self.public_key:
            report.add_step(VerificationStep(
                name="signature",
                status=VerificationStatus.SKIPPED,
                message="공개 키 미제공으로 건너뜀"
            ))
        
        # 3. 타임스탬프 검증
        if self.config.verify_timestamp:
            report.add_step(self._verify_timestamp(token))
        
        # 4. 위험 점수 검증
        if self.config.verify_risk_threshold:
            report.add_step(self._verify_risk_score(token))
        
        return report
    
    def _verify_schema(self, token: PoXToken) -> VerificationStep:
        """스키마 검증"""
        errors = []
        
        # 노드 ID 검증
        if len(token.node_id) != 16:
            errors.append(f"node_id 길이 오류: {len(token.node_id)} (예상: 16)")
        
        # 지문 검증
        fp = token.fingerprint
        if len(fp.feature_vector) != 64:
            errors.append(f"feature_vector 길이 오류: {len(fp.feature_vector)} (예상: 64)")
        if len(fp.correlation_hash) != 32:
            errors.append(f"correlation_hash 길이 오류: {len(fp.correlation_hash)} (예상: 32)")
        if fp.sensor_count < self.config.min_sensor_count:
            errors.append(f"센서 수 부족: {fp.sensor_count} < {self.config.min_sensor_count}")
        
        # 위험 점수 범위
        if not (0 <= token.risk_score <= 100):
            errors.append(f"위험 점수 범위 오류: {token.risk_score}")
        
        if errors:
            return VerificationStep(
                name="schema",
                status=VerificationStatus.FAILED,
                message="; ".join(errors),
                details={"errors": errors}
            )
        
        return VerificationStep(
            name="schema",
            status=VerificationStatus.PASSED,
            message="스키마 유효"
        )
    
    def _verify_signature(self, token: PoXToken) -> VerificationStep:
        """서명 검증"""
        if not token.is_signed:
            return VerificationStep(
                name="signature",
                status=VerificationStatus.FAILED,
                message="토큰에 서명이 없음"
            )
        
        result = self._signature_verifier.verify(token, self.public_key)
        
        if result.is_valid:
            return VerificationStep(
                name="signature",
                status=VerificationStatus.PASSED,
                message="서명 유효"
            )
        else:
            return VerificationStep(
                name="signature",
                status=VerificationStatus.FAILED,
                message=result.error or "서명 검증 실패"
            )
    
    def _verify_timestamp(self, token: PoXToken) -> VerificationStep:
        """타임스탬프 검증"""
        now_ns = int(datetime.now().timestamp() * 1e9)
        token_age_ns = now_ns - token.timestamp_ns
        token_age_seconds = token_age_ns / 1e9
        
        # 미래 시간 검사 (1분 여유)
        if token.timestamp_ns > now_ns + 60_000_000_000:
            return VerificationStep(
                name="timestamp",
                status=VerificationStatus.FAILED,
                message="미래 타임스탬프",
                details={"future_by_ns": token.timestamp_ns - now_ns}
            )
        
        # 수명 검사
        if token_age_seconds > self.config.max_age_seconds:
            return VerificationStep(
                name="timestamp",
                status=VerificationStatus.WARNING,
                message=f"토큰 수명 초과: {token_age_seconds:.0f}s > {self.config.max_age_seconds}s",
                details={"age_seconds": token_age_seconds}
            )
        
        return VerificationStep(
            name="timestamp",
            status=VerificationStatus.PASSED,
            message=f"타임스탬프 유효 (수명: {token_age_seconds:.0f}s)"
        )
    
    def _verify_risk_score(self, token: PoXToken) -> VerificationStep:
        """위험 점수 검증"""
        if token.risk_score > self.config.max_risk_score:
            return VerificationStep(
                name="risk_score",
                status=VerificationStatus.WARNING,
                message=f"높은 위험 점수: {token.risk_score:.2f} > {self.config.max_risk_score}",
                details={"risk_score": token.risk_score}
            )
        
        return VerificationStep(
            name="risk_score",
            status=VerificationStatus.PASSED,
            message=f"위험 점수 정상: {token.risk_score:.2f}"
        )
    
    def quick_check(self, token: PoXToken) -> bool:
        """빠른 유효성 확인 (상세 보고서 없이)"""
        report = self.validate(token)
        return report.is_valid


class VerificationEngine:
    """
    검증 엔진
    
    다중 공개 키를 관리하고 토큰을 검증합니다.
    노드별 키 매핑을 지원합니다.
    
    Example:
        >>> engine = VerificationEngine()
        >>> engine.register_key(node_id, public_key)
        >>> report = engine.verify(token)
    """
    
    def __init__(self, config: VerificationConfig | None = None):
        """
        Args:
            config: 검증 설정
        """
        self.config = config or VerificationConfig()
        self._keys: dict[bytes, bytes] = {}  # node_id -> public_key
    
    def register_key(self, node_id: bytes, public_key: bytes) -> None:
        """노드 공개 키 등록"""
        self._keys[node_id] = public_key
    
    def unregister_key(self, node_id: bytes) -> bool:
        """노드 공개 키 제거"""
        if node_id in self._keys:
            del self._keys[node_id]
            return True
        return False
    
    def get_key(self, node_id: bytes) -> Optional[bytes]:
        """노드 공개 키 조회"""
        return self._keys.get(node_id)
    
    def verify(self, token: PoXToken) -> VerificationReport:
        """
        토큰 검증
        
        노드 ID에 해당하는 공개 키를 자동으로 찾아 검증합니다.
        
        Args:
            token: 검증할 토큰
        
        Returns:
            VerificationReport
        """
        public_key = self._keys.get(token.node_id)
        validator = TokenValidator(public_key=public_key, config=self.config)
        return validator.validate(token)
    
    def verify_batch(self, tokens: list[PoXToken]) -> list[VerificationReport]:
        """토큰 일괄 검증"""
        return [self.verify(token) for token in tokens]
    
    @property
    def registered_nodes(self) -> list[bytes]:
        """등록된 노드 목록"""
        return list(self._keys.keys())


def validate_token(
    token: PoXToken,
    public_key: bytes | None = None
) -> bool:
    """
    편의 함수: 토큰 검증
    
    Args:
        token: 검증할 토큰
        public_key: 서명 검증용 공개 키
    
    Returns:
        유효 여부
    """
    validator = TokenValidator(public_key=public_key)
    return validator.quick_check(token)
