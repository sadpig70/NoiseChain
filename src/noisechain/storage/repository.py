"""
NoiseChain 토큰 저장소

PoXToken의 영속 저장 및 검색을 담당합니다.
SQLite 기반 경량 저장소 구현.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from noisechain.token.schema import PoXToken, NoiseFingerprint


@dataclass
class StorageConfig:
    """저장소 설정"""
    db_path: str = ":memory:"           # 데이터베이스 경로 (":memory:" = 인메모리)
    create_tables: bool = True          # 테이블 자동 생성
    enable_wal: bool = True             # WAL 모드 활성화 (성능)
    page_size: int = 4096               # 페이지 크기


@dataclass
class TokenQuery:
    """토큰 검색 쿼리"""
    node_id: Optional[bytes] = None      # 노드 ID 필터
    start_time: Optional[int] = None     # 시작 시간 (ns)
    end_time: Optional[int] = None       # 종료 시간 (ns)
    min_risk: Optional[float] = None     # 최소 위험 점수
    max_risk: Optional[float] = None     # 최대 위험 점수
    limit: int = 100                     # 최대 결과 수
    offset: int = 0                      # 오프셋 (페이징)


@dataclass
class StorageStats:
    """저장소 통계"""
    total_tokens: int
    total_size_bytes: int
    oldest_timestamp: Optional[int]
    newest_timestamp: Optional[int]
    unique_nodes: int


class TokenRepository:
    """
    토큰 저장소
    
    SQLite 기반으로 PoXToken을 저장하고 검색합니다.
    
    Example:
        >>> repo = TokenRepository()
        >>> repo.save(token)
        >>> tokens = repo.find_by_node(node_id)
    """
    
    # 테이블 스키마
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash BLOB UNIQUE NOT NULL,
        node_id BLOB NOT NULL,
        timestamp_ns INTEGER NOT NULL,
        risk_score REAL NOT NULL,
        data BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_node_id ON tokens(node_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON tokens(timestamp_ns);
    CREATE INDEX IF NOT EXISTS idx_risk_score ON tokens(risk_score);
    """
    
    def __init__(self, config: StorageConfig | None = None):
        """
        Args:
            config: 저장소 설정 (None이면 인메모리)
        """
        self.config = config or StorageConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        
        if self.config.create_tables:
            self._create_tables()
    
    def _connect(self) -> None:
        """데이터베이스 연결"""
        self._conn = sqlite3.connect(
            self.config.db_path,
            check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        
        # WAL 모드 (동시성 향상)
        if self.config.enable_wal and self.config.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        
        # 페이지 크기 설정
        self._conn.execute(f"PRAGMA page_size={self.config.page_size}")
    
    def _create_tables(self) -> None:
        """테이블 생성"""
        self._conn.executescript(self.CREATE_TABLE_SQL)
        self._conn.commit()
    
    def save(self, token: PoXToken) -> int:
        """
        토큰 저장
        
        Args:
            token: 저장할 PoXToken
        
        Returns:
            저장된 레코드 ID
        """
        token_hash = token.compute_hash()
        data = token.to_bytes()
        
        cursor = self._conn.execute(
            """
            INSERT OR REPLACE INTO tokens (hash, node_id, timestamp_ns, risk_score, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token_hash, token.node_id, token.timestamp_ns, token.risk_score, data)
        )
        self._conn.commit()
        
        return cursor.lastrowid
    
    def save_batch(self, tokens: list[PoXToken]) -> int:
        """
        토큰 일괄 저장
        
        Args:
            tokens: 저장할 토큰 리스트
        
        Returns:
            저장된 토큰 수
        """
        data_list = [
            (
                token.compute_hash(),
                token.node_id,
                token.timestamp_ns,
                token.risk_score,
                token.to_bytes()
            )
            for token in tokens
        ]
        
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO tokens (hash, node_id, timestamp_ns, risk_score, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            data_list
        )
        self._conn.commit()
        
        return len(tokens)
    
    def get_by_hash(self, token_hash: bytes) -> Optional[PoXToken]:
        """해시로 토큰 조회"""
        cursor = self._conn.execute(
            "SELECT data FROM tokens WHERE hash = ?",
            (token_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            return PoXToken.from_bytes(row['data'])
        return None
    
    def find_by_node(
        self, 
        node_id: bytes,
        limit: int = 100,
        offset: int = 0
    ) -> list[PoXToken]:
        """노드 ID로 토큰 검색"""
        cursor = self._conn.execute(
            """
            SELECT data FROM tokens 
            WHERE node_id = ?
            ORDER BY timestamp_ns DESC
            LIMIT ? OFFSET ?
            """,
            (node_id, limit, offset)
        )
        
        return [PoXToken.from_bytes(row['data']) for row in cursor.fetchall()]
    
    def find_by_time_range(
        self,
        start_ns: int,
        end_ns: int,
        limit: int = 100
    ) -> list[PoXToken]:
        """시간 범위로 토큰 검색"""
        cursor = self._conn.execute(
            """
            SELECT data FROM tokens 
            WHERE timestamp_ns >= ? AND timestamp_ns <= ?
            ORDER BY timestamp_ns ASC
            LIMIT ?
            """,
            (start_ns, end_ns, limit)
        )
        
        return [PoXToken.from_bytes(row['data']) for row in cursor.fetchall()]
    
    def find_by_risk_range(
        self,
        min_risk: float,
        max_risk: float,
        limit: int = 100
    ) -> list[PoXToken]:
        """위험 점수 범위로 토큰 검색"""
        cursor = self._conn.execute(
            """
            SELECT data FROM tokens 
            WHERE risk_score >= ? AND risk_score <= ?
            ORDER BY risk_score DESC
            LIMIT ?
            """,
            (min_risk, max_risk, limit)
        )
        
        return [PoXToken.from_bytes(row['data']) for row in cursor.fetchall()]
    
    def query(self, q: TokenQuery) -> list[PoXToken]:
        """
        복합 쿼리로 토큰 검색
        
        Args:
            q: 검색 쿼리
        
        Returns:
            조건에 맞는 토큰 리스트
        """
        conditions = []
        params = []
        
        if q.node_id:
            conditions.append("node_id = ?")
            params.append(q.node_id)
        
        if q.start_time is not None:
            conditions.append("timestamp_ns >= ?")
            params.append(q.start_time)
        
        if q.end_time is not None:
            conditions.append("timestamp_ns <= ?")
            params.append(q.end_time)
        
        if q.min_risk is not None:
            conditions.append("risk_score >= ?")
            params.append(q.min_risk)
        
        if q.max_risk is not None:
            conditions.append("risk_score <= ?")
            params.append(q.max_risk)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT data FROM tokens 
            WHERE {where_clause}
            ORDER BY timestamp_ns DESC
            LIMIT ? OFFSET ?
        """
        params.extend([q.limit, q.offset])
        
        cursor = self._conn.execute(sql, params)
        return [PoXToken.from_bytes(row['data']) for row in cursor.fetchall()]
    
    def count(self, q: TokenQuery | None = None) -> int:
        """
        토큰 수 조회
        
        Args:
            q: 검색 쿼리 (None이면 전체)
        
        Returns:
            토큰 수
        """
        if q is None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM tokens")
        else:
            conditions = []
            params = []
            
            if q.node_id:
                conditions.append("node_id = ?")
                params.append(q.node_id)
            
            if q.start_time is not None:
                conditions.append("timestamp_ns >= ?")
                params.append(q.start_time)
            
            if q.end_time is not None:
                conditions.append("timestamp_ns <= ?")
                params.append(q.end_time)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            cursor = self._conn.execute(
                f"SELECT COUNT(*) FROM tokens WHERE {where_clause}",
                params
            )
        
        return cursor.fetchone()[0]
    
    def delete_by_hash(self, token_hash: bytes) -> bool:
        """해시로 토큰 삭제"""
        cursor = self._conn.execute(
            "DELETE FROM tokens WHERE hash = ?",
            (token_hash,)
        )
        self._conn.commit()
        return cursor.rowcount > 0
    
    def delete_by_node(self, node_id: bytes) -> int:
        """노드 ID로 토큰 삭제"""
        cursor = self._conn.execute(
            "DELETE FROM tokens WHERE node_id = ?",
            (node_id,)
        )
        self._conn.commit()
        return cursor.rowcount
    
    def delete_before(self, timestamp_ns: int) -> int:
        """특정 시간 이전 토큰 삭제 (정리용)"""
        cursor = self._conn.execute(
            "DELETE FROM tokens WHERE timestamp_ns < ?",
            (timestamp_ns,)
        )
        self._conn.commit()
        return cursor.rowcount
    
    def get_stats(self) -> StorageStats:
        """저장소 통계 조회"""
        cursor = self._conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(LENGTH(data)) as total_size,
                MIN(timestamp_ns) as oldest,
                MAX(timestamp_ns) as newest,
                COUNT(DISTINCT node_id) as unique_nodes
            FROM tokens
        """)
        row = cursor.fetchone()
        
        return StorageStats(
            total_tokens=row['total'] or 0,
            total_size_bytes=row['total_size'] or 0,
            oldest_timestamp=row['oldest'],
            newest_timestamp=row['newest'],
            unique_nodes=row['unique_nodes'] or 0
        )
    
    def iterate_all(self, batch_size: int = 100) -> Iterator[PoXToken]:
        """모든 토큰 순회 (메모리 효율적)"""
        offset = 0
        while True:
            cursor = self._conn.execute(
                """
                SELECT data FROM tokens 
                ORDER BY timestamp_ns ASC
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset)
            )
            rows = cursor.fetchall()
            
            if not rows:
                break
            
            for row in rows:
                yield PoXToken.from_bytes(row['data'])
            
            offset += batch_size
    
    def vacuum(self) -> None:
        """데이터베이스 최적화 (VACUUM)"""
        self._conn.execute("VACUUM")
    
    def close(self) -> None:
        """연결 종료"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "TokenRepository":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __del__(self) -> None:
        self.close()


def create_repository(
    db_path: str | Path | None = None
) -> TokenRepository:
    """
    편의 함수: 저장소 생성
    
    Args:
        db_path: 데이터베이스 경로 (None이면 인메모리)
    
    Returns:
        TokenRepository
    """
    config = StorageConfig(
        db_path=str(db_path) if db_path else ":memory:"
    )
    return TokenRepository(config)
