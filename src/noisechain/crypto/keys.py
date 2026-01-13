"""
NoiseChain 키 관리

Ed25519 키 쌍 생성 및 관리를 담당합니다.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


@dataclass
class KeyPair:
    """
    Ed25519 키 쌍
    
    Attributes:
        signing_key: 개인 키 (서명용, 32 bytes seed)
        verify_key: 공개 키 (검증용, 32 bytes)
    """
    signing_key: SigningKey
    verify_key: VerifyKey
    
    @classmethod
    def generate(cls) -> "KeyPair":
        """새 키 쌍 생성"""
        signing_key = SigningKey.generate()
        return cls(
            signing_key=signing_key,
            verify_key=signing_key.verify_key
        )
    
    @classmethod
    def from_seed(cls, seed: bytes) -> "KeyPair":
        """
        시드에서 키 쌍 생성 (결정적)
        
        Args:
            seed: 32 bytes 시드
        
        Returns:
            KeyPair
        """
        if len(seed) != 32:
            raise ValueError(f"시드는 32 bytes여야 합니다: {len(seed)}")
        
        signing_key = SigningKey(seed)
        return cls(
            signing_key=signing_key,
            verify_key=signing_key.verify_key
        )
    
    @classmethod
    def from_hex(cls, hex_seed: str) -> "KeyPair":
        """16진수 시드에서 키 쌍 생성"""
        seed = bytes.fromhex(hex_seed)
        return cls.from_seed(seed)
    
    @property
    def seed(self) -> bytes:
        """개인 키 시드 (32 bytes)"""
        return bytes(self.signing_key)
    
    @property
    def seed_hex(self) -> str:
        """개인 키 시드 (16진수)"""
        return self.seed.hex()
    
    @property
    def public_key(self) -> bytes:
        """공개 키 (32 bytes)"""
        return bytes(self.verify_key)
    
    @property
    def public_key_hex(self) -> str:
        """공개 키 (16진수)"""
        return self.public_key.hex()
    
    def save(self, path: str | Path, include_private: bool = True) -> None:
        """
        키 저장
        
        Args:
            path: 저장 경로
            include_private: 개인 키 포함 여부
        """
        path = Path(path)
        
        if include_private:
            # 개인 키 (시드) 저장
            private_path = path.with_suffix('.key')
            private_path.write_bytes(self.seed)
            
        # 공개 키 저장
        public_path = path.with_suffix('.pub')
        public_path.write_bytes(self.public_key)
    
    @classmethod
    def load(cls, path: str | Path) -> "KeyPair":
        """
        키 로드
        
        Args:
            path: 키 파일 경로 (확장자 제외)
        
        Returns:
            KeyPair
        """
        path = Path(path)
        private_path = path.with_suffix('.key')
        
        if not private_path.exists():
            raise FileNotFoundError(f"개인 키 파일을 찾을 수 없습니다: {private_path}")
        
        seed = private_path.read_bytes()
        return cls.from_seed(seed)
    
    @classmethod
    def load_public_only(cls, path: str | Path) -> VerifyKey:
        """
        공개 키만 로드
        
        Args:
            path: 공개 키 파일 경로
        
        Returns:
            VerifyKey
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.pub')
        
        if not path.exists():
            raise FileNotFoundError(f"공개 키 파일을 찾을 수 없습니다: {path}")
        
        return VerifyKey(path.read_bytes())
    
    def __repr__(self) -> str:
        return f"KeyPair(public={self.public_key_hex[:16]}...)"


class KeyManager:
    """
    키 매니저
    
    노드별 키 쌍을 관리하고 캐싱합니다.
    
    Example:
        >>> manager = KeyManager()
        >>> keypair = manager.get_or_create("node1")
        >>> signature = keypair.signing_key.sign(data)
    """
    
    def __init__(self, keys_dir: str | Path | None = None):
        """
        Args:
            keys_dir: 키 저장 디렉토리 (None이면 메모리만)
        """
        self.keys_dir = Path(keys_dir) if keys_dir else None
        self._cache: dict[str, KeyPair] = {}
    
    def get_or_create(self, node_id: str) -> KeyPair:
        """
        노드 키 조회 또는 생성
        
        Args:
            node_id: 노드 식별자
        
        Returns:
            KeyPair
        """
        # 캐시 확인
        if node_id in self._cache:
            return self._cache[node_id]
        
        # 디스크에서 로드 시도
        if self.keys_dir:
            key_path = self.keys_dir / node_id
            try:
                keypair = KeyPair.load(key_path)
                self._cache[node_id] = keypair
                return keypair
            except FileNotFoundError:
                pass
        
        # 새로 생성
        keypair = KeyPair.generate()
        self._cache[node_id] = keypair
        
        # 디스크에 저장
        if self.keys_dir:
            self.keys_dir.mkdir(parents=True, exist_ok=True)
            keypair.save(self.keys_dir / node_id)
        
        return keypair
    
    def get(self, node_id: str) -> Optional[KeyPair]:
        """노드 키 조회 (없으면 None)"""
        if node_id in self._cache:
            return self._cache[node_id]
        
        if self.keys_dir:
            key_path = self.keys_dir / node_id
            try:
                keypair = KeyPair.load(key_path)
                self._cache[node_id] = keypair
                return keypair
            except FileNotFoundError:
                pass
        
        return None
    
    def register(self, node_id: str, keypair: KeyPair) -> None:
        """키 등록"""
        self._cache[node_id] = keypair
        
        if self.keys_dir:
            self.keys_dir.mkdir(parents=True, exist_ok=True)
            keypair.save(self.keys_dir / node_id)
    
    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
    
    @property
    def cached_nodes(self) -> list[str]:
        """캐시된 노드 목록"""
        return list(self._cache.keys())
