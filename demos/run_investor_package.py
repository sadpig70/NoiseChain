import sys
import os
import json
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from noisechain import create_pipeline, PipelineConfig
from noisechain.demo import cmd_benchmark, cmd_demo

def run_investor_demo():
    print("🚀 NoiseChain Investor Demo & Benchmark Script")
    print("============================================")
    
    # 결과 저장 디렉토리
    output_dir = Path("demos/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 전체 데모 실행 (기능 검증)
    print("\n[PART 1] Running End-to-End Functional Demo...")
    demo_args = type('Args', (), {'command': 'demo'})()
    cmd_demo(demo_args)
    
    # 2. 벤치마크 실행 (성능 검증)
    print("\n[PART 2] Running Performance Benchmark (100 Iterations)...")
    bench_file = output_dir / f"benchmark_report_{timestamp}.json"
    
    bench_args = type('Args', (), {
        'command': 'benchmark',
        'iterations': 100,
        'samples': 256,
        'output': str(bench_file)
    })()
    
    cmd_benchmark(bench_args)
    
    print("\n============================================")
    print(f"✅ All tests completed.")
    print(f"📝 Benchmark Report saved to: {bench_file}")
    print("============================================")

if __name__ == "__main__":
    run_investor_demo()
