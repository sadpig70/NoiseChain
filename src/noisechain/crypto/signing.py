"""
NoiseChain 서명 모듈

Ed25519를 사용한 PoXToken 서명 및 검증을 담당합니다.
"""

from dataclasses import dataclass
from typing import Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError

from noisechain.token.schema import PoXToken
from noisechain.crypto.keys import KeyPair


@dataclass
class SignatureResult:
    """서명 결과"""
    signature: bytes          # 64 bytes Ed25519 signature
    public_key: bytes         # 32 bytes 서명자 공개 키
    signed_token: PoXToken    # 서명된 토큰


@dataclass 
class VerificationResult:
    """검증 결과"""
    is_valid: bool
    error: Optional[str] = None
    public_key: Optional[bytes] = None


class TokenSigner:
    """
    토큰 서명기
    
    Ed25519를 사용하여 PoXToken에 서명합니다.
    
    Example:
        >>> keypair = KeyPair.generate()
        >>> signer = TokenSigner(keypair)
        >>> result = signer.sign(token)
        >>> print(result.signed_token.is_signed)
        True
    """
    
    def __init__(self, keypair: KeyPair):
        """
        Args:
            keypair: 서명에 사용할 키 쌍
        """
        self.keypair = keypair
    
    def sign(self, token: PoXToken) -> SignatureResult:
        """
        토큰 서명
        
        Args:
            token: 서명할 PoXToken
        
        Returns:
            SignatureResult (서명, 공개 키, 서명된 토큰)
        """
        # 서명 대상 데이터
        data = token.get_signing_data()
        
        # Ed25519 서명 생성
        signed = self.keypair.signing_key.sign(data)
        signature = signed.signature  # 64 bytes
        
        # 서명된 토큰 생성 (새 인스턴스)
        signed_token = PoXToken(
            version=token.version,
            token_type=token.token_type,
            timestamp_ns=token.timestamp_ns,
            node_id=token.node_id,
            fingerprint=token.fingerprint,
            risk_score=token.risk_score,
            nonce=token.nonce,
            signature=signature,
            metadata=token.metadata.copy()
        )
        
        return SignatureResult(
            signature=signature,
            public_key=self.keypair.public_key,
            signed_token=signed_token
        )
    
    def sign_in_place(self, token: PoXToken) -> bytes:
        """
        토큰에 직접 서명 (in-place)
        
        Args:
            token: 서명할 PoXToken (수정됨)
        
        Returns:
            서명 바이트
        """
        data = token.get_signing_data()
        signed = self.keypair.signing_key.sign(data)
        
        # 토큰 객체에 직접 서명 설정
        object.__setattr__(token, 'signature', signed.signature)
        
        return signed.signature


class TokenVerifier:
    """
    토큰 검증기
    
    Ed25519 서명을 검증합니다.
    
    Example:
        >>> verifier = TokenVerifier()
        >>> result = verifier.verify(token, public_key)
        >>> print(result.is_valid)
    """
    
    def verify(
        self, 
        token: PoXToken, 
        public_key: bytes | VerifyKey
    ) -> VerificationResult:
        """
        토큰 서명 검증
        
        Args:
            token: 검증할 PoXToken
            public_key: 서명자 공개 키 (32 bytes 또는 VerifyKey)
        
        Returns:
            VerificationResult
        """
        # 서명 확인
        if not token.is_signed:
            return VerificationResult(
                is_valid=False,
                error="토큰에 서명이 없습니다"
            )
        
        # VerifyKey 변환
        if isinstance(public_key, bytes):
            if len(public_key) != 32:
                return VerificationResult(
                    is_valid=False,
                    error=f"잘못된 공개 키 길이: {len(public_key)}"
                )
            verify_key = VerifyKey(public_key)
        else:
            verify_key = public_key
        
        # 서명 대상 데이터
        data = token.get_signing_data()
        
        try:
            # Ed25519 서명 검증
            verify_key.verify(data, token.signature)
            
            return VerificationResult(
                is_valid=True,
                public_key=bytes(verify_key)
            )
        except BadSignatureError as e:
            return VerificationResult(
                is_valid=False,
                error=f"서명 검증 실패: {e}"
            )
        except Exception as e:
            return VerificationResult(
                is_valid=False,
                error=f"검증 오류: {e}"
            )
    
    def verify_with_keypair(
        self, 
        token: PoXToken, 
        keypair: KeyPair
    ) -> VerificationResult:
        """키 쌍으로 검증 (편의 메서드)"""
        return self.verify(token, keypair.verify_key)


def sign_token(token: PoXToken, keypair: KeyPair) -> PoXToken:
    """
    편의 함수: 토큰 서명
    
    Args:
        token: 서명할 토큰
        keypair: 키 쌍
    
    Returns:
        서명된 토큰
    """
    signer = TokenSigner(keypair)
    return signer.sign(token).signed_token


def verify_token(token: PoXToken, public_key: bytes) -> bool:
    """
    편의 함수: 토큰 검증
    
    Args:
        token: 검증할 토큰
        public_key: 공개 키
    
    Returns:
        유효 여부
    """
    verifier = TokenVerifier()
    return verifier.verify(token, public_key).is_valid
