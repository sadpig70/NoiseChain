"""
NoiseChain E2E 파이프라인 테스트

NoiseChainPipeline 통합 테스트입니다.
"""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from noisechain import NoiseChainPipeline, PipelineConfig, create_pipeline
from noisechain.engine import PipelineResult


# ============================================================
# PipelineConfig 테스트
# ============================================================

class TestPipelineConfig:
    """PipelineConfig 테스트"""
    
    def test_default_values(self):
        """기본값 확인"""
        config = PipelineConfig()
        
        assert len(config.node_id) == 16
        assert config.sample_count == 256
        assert config.db_path == ":memory:"
    
    def test_custom_values(self):
        """사용자 정의 값"""
        node_id = uuid.uuid4().bytes
        config = PipelineConfig(
            node_id=node_id,
            sample_count=512,
            max_risk_score=50.0
        )
        
        assert config.node_id == node_id
        assert config.sample_count == 512
        assert config.max_risk_score == 50.0


# ============================================================
# PipelineResult 테스트
# ============================================================

class TestPipelineResult:
    """PipelineResult 테스트"""
    
    def test_success_result(self):
        """성공 결과"""
        result = PipelineResult(success=True)
        
        assert result.success is True
        assert result.error is None
    
    def test_failure_result(self):
        """실패 결과"""
        result = PipelineResult(success=False, error="Test error")
        
        assert result.success is False
        assert result.error == "Test error"
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        result = PipelineResult(success=True)
        
        d = result.to_dict()
        
        assert "success" in d
        assert d["success"] is True


# ============================================================
# NoiseChainPipeline 테스트
# ============================================================

class TestNoiseChainPipeline:
    """NoiseChainPipeline 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        with NoiseChainPipeline() as pipeline:
            assert len(pipeline.node_id) == 16
            assert len(pipeline.public_key) == 32
    
    def test_custom_config(self):
        """사용자 정의 설정"""
        node_id = uuid.uuid4().bytes
        config = PipelineConfig(node_id=node_id)
        
        with NoiseChainPipeline(config) as pipeline:
            assert pipeline.node_id == node_id
    
    def test_collect_samples(self):
        """센서 샘플 수집"""
        with NoiseChainPipeline() as pipeline:
            samples = pipeline.collect_samples(count=50)
            
            assert len(samples) > 0
            for name, data in samples.items():
                assert len(data) == 50
    
    def test_generate_token(self):
        """토큰 생성"""
        with NoiseChainPipeline() as pipeline:
            # 직접 데이터 제공
            sensor_data = {
                "sensor1": np.random.randn(100),
                "sensor2": np.random.randn(100),
            }
            
            token = pipeline.generate_token(sensor_data)
            
            assert token is not None
            assert token.is_signed
            assert token.node_id == pipeline.node_id
    
    def test_generate_and_store(self):
        """토큰 생성 및 저장"""
        with NoiseChainPipeline() as pipeline:
            sensor_data = {
                "sensor1": np.random.randn(100),
            }
            
            result = pipeline.generate_and_store(sensor_data)
            
            assert result.success is True
            assert result.token is not None
            assert result.verification is not None
            assert result.verification.is_valid is True
    
    def test_verify_token(self):
        """토큰 검증"""
        with NoiseChainPipeline() as pipeline:
            sensor_data = {"sensor1": np.random.randn(100)}
            token = pipeline.generate_token(sensor_data)
            
            report = pipeline.verify_token(token)
            
            assert report.is_valid is True
    
    def test_verify_by_hash(self):
        """해시로 토큰 검증"""
        with NoiseChainPipeline() as pipeline:
            sensor_data = {"sensor1": np.random.randn(100)}
            result = pipeline.generate_and_store(sensor_data)
            token_hash = result.token.compute_hash()
            
            report = pipeline.verify_by_hash(token_hash)
            
            assert report is not None
            assert report.is_valid is True
    
    def test_get_recent_tokens(self):
        """최근 토큰 조회"""
        with NoiseChainPipeline() as pipeline:
            # 5개 토큰 생성
            for _ in range(5):
                pipeline.generate_and_store({"s": np.random.randn(50)})
            
            tokens = pipeline.get_recent_tokens(limit=3)
            
            assert len(tokens) == 3
    
    def test_get_stats(self):
        """통계 조회"""
        with NoiseChainPipeline() as pipeline:
            pipeline.generate_and_store({"s": np.random.randn(50)})
            
            stats = pipeline.get_stats()
            
            assert stats["total_tokens"] == 1
            assert "node_id" in stats


# ============================================================
# create_pipeline 편의 함수 테스트
# ============================================================

class TestCreatePipeline:
    """create_pipeline 편의 함수 테스트"""
    
    def test_memory_pipeline(self):
        """인메모리 파이프라인"""
        with create_pipeline() as pipeline:
            result = pipeline.generate_and_store({"s": np.random.randn(50)})
            
            assert result.success is True
    
    def test_file_pipeline(self):
        """파일 저장소 파이프라인"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            with create_pipeline(db_path=db_path) as pipeline:
                pipeline.generate_and_store({"s": np.random.randn(50)})
            
            # 파일 생성 확인
            assert db_path.exists()


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegrationPipeline:
    """파이프라인 통합 테스트"""
    
    def test_full_workflow(self):
        """전체 워크플로우: 수집 → 생성 → 서명 → 저장 → 검증"""
        with NoiseChainPipeline() as pipeline:
            # 1. 여러 토큰 생성 및 저장
            results = []
            for i in range(3):
                sensor_data = {
                    "cpu_temp": 50 + 10 * np.sin(np.linspace(0, 2*np.pi, 100)),
                    "entropy": np.random.randint(0, 256, 100).astype(float),
                }
                result = pipeline.generate_and_store(sensor_data)
                results.append(result)
            
            # 2. 전부 성공 확인
            assert all(r.success for r in results)
            
            # 3. 모든 토큰 검증 통과 확인
            assert all(r.verification.is_valid for r in results)
            
            # 4. 저장소 통계 확인
            stats = pipeline.get_stats()
            assert stats["total_tokens"] == 3
            
            # 5. 최근 토큰 조회
            recent = pipeline.get_recent_tokens(limit=10)
            assert len(recent) == 3
            
            # 6. 해시로 개별 검증
            for result in results:
                token_hash = result.token.compute_hash()
                report = pipeline.verify_by_hash(token_hash)
                assert report.is_valid is True
    
    def test_persistence(self):
        """영속성 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persistence_test.db"
            node_id = uuid.uuid4().bytes
            
            # 1. 첫 번째 세션: 토큰 생성
            config = PipelineConfig(node_id=node_id, db_path=str(db_path))
            with NoiseChainPipeline(config) as pipeline1:
                result = pipeline1.generate_and_store({"s": np.random.randn(50)})
                token_hash = result.token.compute_hash()
            
            # 2. 두 번째 세션: 토큰 조회
            config2 = PipelineConfig(node_id=node_id, db_path=str(db_path))
            with NoiseChainPipeline(config2) as pipeline2:
                retrieved = pipeline2.repository.get_by_hash(token_hash)
                
                assert retrieved is not None
                assert retrieved.node_id == node_id
