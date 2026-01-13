"""
NoiseChain 암호화 모듈 테스트

KeyPair, KeyManager, TokenSigner, TokenVerifier 단위 테스트입니다.
"""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from noisechain.crypto.keys import KeyManager, KeyPair
from noisechain.crypto.signing import (
    SignatureResult,
    TokenSigner,
    TokenVerifier,
    sign_token,
    verify_token,
)
from noisechain.token.builder import TokenBuilder
from noisechain.token.schema import NoiseFingerprint, PoXToken


# ============================================================
# KeyPair 테스트
# ============================================================

class TestKeyPair:
    """KeyPair 테스트"""
    
    def test_generate(self):
        """키 생성"""
        keypair = KeyPair.generate()
        
        assert len(keypair.seed) == 32
        assert len(keypair.public_key) == 32
    
    def test_from_seed_deterministic(self):
        """시드에서 결정적 키 생성"""
        seed = bytes(range(32))
        
        kp1 = KeyPair.from_seed(seed)
        kp2 = KeyPair.from_seed(seed)
        
        assert kp1.seed == kp2.seed
        assert kp1.public_key == kp2.public_key
    
    def test_from_seed_invalid_length(self):
        """잘못된 시드 길이"""
        with pytest.raises(ValueError):
            KeyPair.from_seed(bytes(16))
    
    def test_from_hex(self):
        """16진수에서 키 생성"""
        hex_seed = "00" * 32
        
        keypair = KeyPair.from_hex(hex_seed)
        
        assert keypair.seed == bytes(32)
    
    def test_seed_hex_property(self):
        """시드 16진수 속성"""
        keypair = KeyPair.from_seed(bytes(32))
        
        assert keypair.seed_hex == "00" * 32
    
    def test_public_key_hex_property(self):
        """공개 키 16진수 속성"""
        keypair = KeyPair.generate()
        
        assert len(keypair.public_key_hex) == 64  # 32 bytes = 64 hex chars
    
    def test_save_and_load(self):
        """저장 및 로드"""
        with tempfile.TemporaryDirectory() as tmpdir:
            keypair = KeyPair.generate()
            key_path = Path(tmpdir) / "test_key"
            
            # 저장
            keypair.save(key_path)
            
            # 로드
            loaded = KeyPair.load(key_path)
            
            assert loaded.seed == keypair.seed
            assert loaded.public_key == keypair.public_key
    
    def test_load_nonexistent(self):
        """존재하지 않는 키 로드"""
        with pytest.raises(FileNotFoundError):
            KeyPair.load("/nonexistent/path")


# ============================================================
# KeyManager 테스트
# ============================================================

class TestKeyManager:
    """KeyManager 테스트"""
    
    def test_memory_only(self):
        """메모리 전용 모드"""
        manager = KeyManager()
        
        keypair = manager.get_or_create("node1")
        
        assert keypair is not None
        assert len(manager.cached_nodes) == 1
    
    def test_get_existing(self):
        """기존 키 조회"""
        manager = KeyManager()
        
        kp1 = manager.get_or_create("node1")
        kp2 = manager.get_or_create("node1")
        
        assert kp1.seed == kp2.seed
    
    def test_get_nonexistent(self):
        """존재하지 않는 키 조회"""
        manager = KeyManager()
        
        result = manager.get("nonexistent")
        
        assert result is None
    
    def test_register(self):
        """키 등록"""
        manager = KeyManager()
        keypair = KeyPair.generate()
        
        manager.register("custom_node", keypair)
        
        retrieved = manager.get("custom_node")
        assert retrieved.seed == keypair.seed
    
    def test_clear_cache(self):
        """캐시 초기화"""
        manager = KeyManager()
        manager.get_or_create("node1")
        
        manager.clear_cache()
        
        assert len(manager.cached_nodes) == 0
    
    def test_with_directory(self):
        """디렉토리 저장 모드"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(keys_dir=tmpdir)
            
            keypair = manager.get_or_create("node1")
            
            # 파일 확인
            key_file = Path(tmpdir) / "node1.key"
            assert key_file.exists()


# ============================================================
# TokenSigner 테스트
# ============================================================

class TestTokenSigner:
    """TokenSigner 테스트"""
    
    @pytest.fixture
    def keypair(self):
        """테스트용 키 쌍"""
        return KeyPair.generate()
    
    @pytest.fixture
    def token(self):
        """테스트용 토큰"""
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
        return PoXToken(
            node_id=uuid.uuid4().bytes,
            fingerprint=fingerprint,
            risk_score=10.0
        )
    
    def test_sign(self, keypair, token):
        """토큰 서명"""
        signer = TokenSigner(keypair)
        
        result = signer.sign(token)
        
        assert len(result.signature) == 64
        assert result.signed_token.is_signed
        assert result.public_key == keypair.public_key
    
    def test_sign_creates_new_token(self, keypair, token):
        """서명이 새 토큰을 생성"""
        signer = TokenSigner(keypair)
        
        result = signer.sign(token)
        
        # 원본은 서명되지 않음
        assert not token.is_signed
        # 결과 토큰은 서명됨
        assert result.signed_token.is_signed
    
    def test_sign_deterministic(self, keypair, token):
        """동일 데이터에 동일 서명"""
        signer = TokenSigner(keypair)
        
        result1 = signer.sign(token)
        result2 = signer.sign(token)
        
        assert result1.signature == result2.signature


# ============================================================
# TokenVerifier 테스트
# ============================================================

class TestTokenVerifier:
    """TokenVerifier 테스트"""
    
    @pytest.fixture
    def keypair(self):
        """테스트용 키 쌍"""
        return KeyPair.generate()
    
    @pytest.fixture
    def signed_token(self, keypair):
        """서명된 테스트용 토큰"""
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
        token = PoXToken(
            node_id=uuid.uuid4().bytes,
            fingerprint=fingerprint,
            risk_score=10.0
        )
        
        signer = TokenSigner(keypair)
        return signer.sign(token).signed_token
    
    def test_verify_valid(self, keypair, signed_token):
        """유효한 서명 검증"""
        verifier = TokenVerifier()
        
        result = verifier.verify(signed_token, keypair.public_key)
        
        assert result.is_valid is True
        assert result.error is None
    
    def test_verify_wrong_key(self, signed_token):
        """잘못된 키로 검증"""
        other_keypair = KeyPair.generate()
        verifier = TokenVerifier()
        
        result = verifier.verify(signed_token, other_keypair.public_key)
        
        assert result.is_valid is False
        assert "실패" in result.error
    
    def test_verify_unsigned_token(self):
        """서명 없는 토큰 검증"""
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=1,
            sample_count=1
        )
        unsigned_token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint
        )
        
        verifier = TokenVerifier()
        result = verifier.verify(unsigned_token, bytes(32))
        
        assert result.is_valid is False
        assert "서명이 없습니다" in result.error
    
    def test_verify_with_keypair(self, keypair, signed_token):
        """키 쌍으로 검증"""
        verifier = TokenVerifier()
        
        result = verifier.verify_with_keypair(signed_token, keypair)
        
        assert result.is_valid is True


# ============================================================
# 편의 함수 테스트
# ============================================================

class TestConvenienceFunctions:
    """편의 함수 테스트"""
    
    def test_sign_token(self):
        """sign_token 함수"""
        keypair = KeyPair.generate()
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=1,
            sample_count=1
        )
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint
        )
        
        signed = sign_token(token, keypair)
        
        assert signed.is_signed
    
    def test_verify_token(self):
        """verify_token 함수"""
        keypair = KeyPair.generate()
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=1,
            sample_count=1
        )
        token = PoXToken(
            node_id=bytes(16),
            fingerprint=fingerprint
        )
        
        signed = sign_token(token, keypair)
        
        assert verify_token(signed, keypair.public_key) is True


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationCrypto:
    """암호화 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 키 생성 → 토큰 생성 → 서명 → 검증"""
        # 1. 키 생성
        keypair = KeyPair.generate()
        
        # 2. 간단한 토큰 생성 (양자화 문제 회피)
        fingerprint = NoiseFingerprint(
            feature_vector=bytes(64),
            correlation_hash=bytes(32),
            sensor_count=4,
            sample_count=256
        )
        token = PoXToken(
            node_id=uuid.uuid4().bytes,
            fingerprint=fingerprint,
            risk_score=25.0  # 정수값으로 양자화 손실 없음
        )
        
        # 3. 서명
        signer = TokenSigner(keypair)
        result = signer.sign(token)
        signed_token = result.signed_token
        
        assert signed_token.is_signed
        assert signed_token.size == 199  # 서명 포함
        
        # 4. 서명된 토큰 검증
        verifier = TokenVerifier()
        verify_result = verifier.verify(signed_token, keypair.public_key)
        assert verify_result.is_valid is True
        
        # 5. 잘못된 키로 검증 실패 확인
        other_keypair = KeyPair.generate()
        fail_result = verifier.verify(signed_token, other_keypair.public_key)
        assert fail_result.is_valid is False
