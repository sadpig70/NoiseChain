"""
NoiseChain 저장소 모듈 테스트

TokenRepository 및 관련 클래스의 단위 테스트입니다.
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from noisechain.storage.repository import (
    StorageConfig,
    StorageStats,
    TokenQuery,
    TokenRepository,
    create_repository,
)
from noisechain.token.schema import NoiseFingerprint, PoXToken


# ============================================================
# 테스트 헬퍼
# ============================================================

def create_test_token(
    node_id: bytes | None = None,
    timestamp_ns: int | None = None,
    risk_score: float = 10.0
) -> PoXToken:
    """테스트용 토큰 생성"""
    fingerprint = NoiseFingerprint(
        feature_vector=bytes(64),
        correlation_hash=bytes(32),
        sensor_count=4,
        sample_count=256
    )
    return PoXToken(
        node_id=node_id or uuid.uuid4().bytes,
        fingerprint=fingerprint,
        risk_score=risk_score,
        timestamp_ns=timestamp_ns or 1700000000_000000000
    )


# ============================================================
# StorageConfig 테스트
# ============================================================

class TestStorageConfig:
    """StorageConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        config = StorageConfig()
        
        assert config.db_path == ":memory:"
        assert config.create_tables is True
        assert config.enable_wal is True
    
    def test_custom_values(self):
        """사용자 정의 값"""
        config = StorageConfig(
            db_path="/tmp/test.db",
            enable_wal=False
        )
        
        assert config.db_path == "/tmp/test.db"
        assert config.enable_wal is False


# ============================================================
# TokenRepository 테스트
# ============================================================

class TestTokenRepository:
    """TokenRepository 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화 (인메모리)"""
        repo = TokenRepository()
        
        assert repo.count() == 0
        repo.close()
    
    def test_save_and_get(self):
        """저장 및 조회"""
        repo = TokenRepository()
        token = create_test_token()
        
        # 저장
        row_id = repo.save(token)
        assert row_id > 0
        
        # 해시로 조회
        token_hash = token.compute_hash()
        retrieved = repo.get_by_hash(token_hash)
        
        assert retrieved is not None
        assert retrieved.node_id == token.node_id
        
        repo.close()
    
    def test_save_duplicate_updates(self):
        """중복 저장 시 업데이트"""
        repo = TokenRepository()
        token = create_test_token(risk_score=10.0)
        
        repo.save(token)
        repo.save(token)  # 동일 해시
        
        assert repo.count() == 1
        repo.close()
    
    def test_save_batch(self):
        """배치 저장"""
        repo = TokenRepository()
        tokens = [create_test_token() for _ in range(10)]
        
        count = repo.save_batch(tokens)
        
        assert count == 10
        assert repo.count() == 10
        repo.close()
    
    def test_find_by_node(self):
        """노드 ID로 검색"""
        repo = TokenRepository()
        node_id = uuid.uuid4().bytes
        
        # 같은 노드의 토큰 3개
        for i in range(3):
            token = create_test_token(
                node_id=node_id,
                timestamp_ns=1700000000_000000000 + i
            )
            repo.save(token)
        
        # 다른 노드의 토큰 2개
        for _ in range(2):
            repo.save(create_test_token())
        
        found = repo.find_by_node(node_id)
        
        assert len(found) == 3
        repo.close()
    
    def test_find_by_time_range(self):
        """시간 범위로 검색"""
        repo = TokenRepository()
        
        # 다양한 시간의 토큰
        for i in range(5):
            token = create_test_token(
                timestamp_ns=1700000000_000000000 + i * 1000000000
            )
            repo.save(token)
        
        # 중간 범위 검색
        found = repo.find_by_time_range(
            start_ns=1700000001_000000000,
            end_ns=1700000003_000000000
        )
        
        assert len(found) == 3
        repo.close()
    
    def test_find_by_risk_range(self):
        """위험 점수 범위로 검색"""
        repo = TokenRepository()
        
        # 다양한 위험 점수
        for i in range(5):
            token = create_test_token(risk_score=i * 20.0)
            repo.save(token)
        
        # 고위험 검색 (40-80)
        found = repo.find_by_risk_range(min_risk=40.0, max_risk=80.0)
        
        assert len(found) == 3  # 40, 60, 80
        repo.close()
    
    def test_query_combined(self):
        """복합 쿼리"""
        repo = TokenRepository()
        node_id = uuid.uuid4().bytes
        
        # 테스트 데이터
        for i in range(10):
            token = create_test_token(
                node_id=node_id if i < 5 else uuid.uuid4().bytes,
                timestamp_ns=1700000000_000000000 + i * 1000000000,
                risk_score=i * 10.0
            )
            repo.save(token)
        
        # 복합 쿼리
        q = TokenQuery(
            node_id=node_id,
            min_risk=20.0,
            limit=10
        )
        found = repo.query(q)
        
        assert len(found) == 3  # 노드 일치 + 위험점수 >= 20 (i=2,3,4)
        repo.close()
    
    def test_count(self):
        """토큰 수 조회"""
        repo = TokenRepository()
        
        for _ in range(5):
            repo.save(create_test_token())
        
        assert repo.count() == 5
        repo.close()
    
    def test_count_with_query(self):
        """쿼리와 함께 토큰 수 조회"""
        repo = TokenRepository()
        node_id = uuid.uuid4().bytes
        
        for i in range(10):
            token = create_test_token(
                node_id=node_id if i < 7 else uuid.uuid4().bytes
            )
            repo.save(token)
        
        q = TokenQuery(node_id=node_id)
        assert repo.count(q) == 7
        repo.close()
    
    def test_delete_by_hash(self):
        """해시로 삭제"""
        repo = TokenRepository()
        token = create_test_token()
        repo.save(token)
        
        deleted = repo.delete_by_hash(token.compute_hash())
        
        assert deleted is True
        assert repo.count() == 0
        repo.close()
    
    def test_delete_by_node(self):
        """노드 ID로 삭제"""
        repo = TokenRepository()
        node_id = uuid.uuid4().bytes
        
        for _ in range(3):
            repo.save(create_test_token(node_id=node_id))
        repo.save(create_test_token())  # 다른 노드
        
        deleted = repo.delete_by_node(node_id)
        
        assert deleted == 3
        assert repo.count() == 1
        repo.close()
    
    def test_delete_before(self):
        """시간 이전 삭제"""
        repo = TokenRepository()
        
        for i in range(5):
            token = create_test_token(
                timestamp_ns=1700000000_000000000 + i * 1000000000
            )
            repo.save(token)
        
        deleted = repo.delete_before(1700000002_000000000)
        
        assert deleted == 2
        assert repo.count() == 3
        repo.close()
    
    def test_get_stats(self):
        """통계 조회"""
        repo = TokenRepository()
        
        for i in range(5):
            token = create_test_token(
                timestamp_ns=1700000000_000000000 + i * 1000000000
            )
            repo.save(token)
        
        stats = repo.get_stats()
        
        assert stats.total_tokens == 5
        assert stats.total_size_bytes > 0
        assert stats.oldest_timestamp == 1700000000_000000000
        assert stats.newest_timestamp == 1700000004_000000000
        repo.close()
    
    def test_iterate_all(self):
        """전체 순회"""
        repo = TokenRepository()
        
        for _ in range(15):
            repo.save(create_test_token())
        
        count = 0
        for token in repo.iterate_all(batch_size=5):
            count += 1
            assert token is not None
        
        assert count == 15
        repo.close()
    
    def test_context_manager(self):
        """컨텍스트 매니저"""
        with TokenRepository() as repo:
            repo.save(create_test_token())
            assert repo.count() == 1


# ============================================================
# 편의 함수 테스트
# ============================================================

class TestCreateRepository:
    """create_repository 편의 함수 테스트"""
    
    def test_memory_repository(self):
        """인메모리 저장소"""
        repo = create_repository()
        
        repo.save(create_test_token())
        assert repo.count() == 1
        
        repo.close()
    
    def test_file_repository(self):
        """파일 저장소"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # 저장
            repo1 = create_repository(db_path)
            repo1.save(create_test_token())
            repo1.close()
            
            # 다시 열기
            repo2 = create_repository(db_path)
            assert repo2.count() == 1
            repo2.close()


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationStorage:
    """저장소 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우"""
        with TokenRepository() as repo:
            # 1. 여러 노드의 토큰 저장
            nodes = [uuid.uuid4().bytes for _ in range(3)]
            
            for i, node_id in enumerate(nodes):
                for j in range(5):
                    token = create_test_token(
                        node_id=node_id,
                        timestamp_ns=1700000000_000000000 + j * 1000000000,
                        risk_score=(i + 1) * 10.0
                    )
                    repo.save(token)
            
            # 2. 통계 확인
            stats = repo.get_stats()
            assert stats.total_tokens == 15
            assert stats.unique_nodes == 3
            
            # 3. 노드별 검색
            found = repo.find_by_node(nodes[0])
            assert len(found) == 5
            
            # 4. 고위험 토큰 검색 (risk >= 20)
            high_risk = repo.find_by_risk_range(min_risk=20.0, max_risk=100.0)
            assert len(high_risk) == 10  # 노드 2,3 (risk 20, 30)
            
            # 5. 오래된 토큰 정리
            deleted = repo.delete_before(1700000002_000000000)
            assert deleted == 6  # 각 노드에서 2개씩
            
            # 6. 최종 상태
            assert repo.count() == 9
