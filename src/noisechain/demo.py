#!/usr/bin/env python
"""
NoiseChain CLI 데모

E2E 파이프라인을 시연하는 커맨드라인 인터페이스입니다.

사용법:
    python -m noisechain.demo generate      # 토큰 생성
    python -m noisechain.demo verify <hash> # 토큰 검증
    python -m noisechain.demo stats         # 통계 조회
    python -m noisechain.demo benchmark     # 성능 벤치마크
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from noisechain import NoiseChainPipeline, PipelineConfig, create_pipeline


def print_banner():
    """배너 출력"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                     NoiseChain MVP                        ║
║           Physical Trust Verification Network             ║
╚═══════════════════════════════════════════════════════════╝
    """)


def cmd_generate(args):
    """토큰 생성 명령"""
    print("\n📡 센서 데이터 수집 중...")
    
    config = PipelineConfig(
        db_path=args.db if args.db else ":memory:",
        sample_count=args.samples
    )
    
    with NoiseChainPipeline(config) as pipeline:
        # 토큰 생성
        start = time.perf_counter()
        result = pipeline.generate_and_store()
        elapsed = time.perf_counter() - start
        
        if result.success:
            token = result.token
            report = result.verification
            
            print(f"\n✅ 토큰 생성 성공! ({elapsed*1000:.1f}ms)")
            print(f"\n📋 토큰 정보:")
            print(f"   해시: {token.compute_hash().hex()[:32]}...")
            print(f"   노드 ID: {token.node_id_hex[:16]}...")
            print(f"   위험 점수: {token.risk_score:.2f}")
            print(f"   크기: {token.size} bytes")
            print(f"   서명됨: {'예' if token.is_signed else '아니오'}")
            
            print(f"\n🔍 검증 결과:")
            print(f"   유효: {'✅ 예' if report.is_valid else '❌ 아니오'}")
            print(f"   통과: {report.passed_count}개")
            print(f"   실패: {report.failed_count}개")
            
            for step in report.steps:
                icon = "✅" if step.passed else "❌"
                print(f"   {icon} {step.name}: {step.message}")
        else:
            print(f"\n❌ 토큰 생성 실패: {result.error}")
            return 1
    
    return 0


def cmd_verify(args):
    """토큰 검증 명령"""
    print(f"\n🔍 토큰 검증 중: {args.hash[:32]}...")
    
    try:
        token_hash = bytes.fromhex(args.hash)
    except ValueError:
        print("❌ 잘못된 해시 형식입니다.")
        return 1
    
    config = PipelineConfig(db_path=args.db if args.db else ":memory:")
    
    with NoiseChainPipeline(config) as pipeline:
        report = pipeline.verify_by_hash(token_hash)
        
        if report is None:
            print("❌ 토큰을 찾을 수 없습니다.")
            return 1
        
        print(f"\n📋 검증 결과:")
        print(f"   유효: {'✅ 예' if report.is_valid else '❌ 아니오'}")
        
        for step in report.steps:
            icon = "✅" if step.passed else "❌"
            print(f"   {icon} {step.name}: {step.message}")
    
    return 0


def cmd_stats(args):
    """통계 조회 명령"""
    print("\n📊 저장소 통계 조회 중...")
    
    config = PipelineConfig(db_path=args.db if args.db else ":memory:")
    
    with NoiseChainPipeline(config) as pipeline:
        stats = pipeline.get_stats()
        
        print(f"\n📋 통계:")
        print(f"   노드 ID: {stats['node_id'][:16]}...")
        print(f"   총 토큰 수: {stats['total_tokens']}")
        print(f"   총 크기: {stats['total_size_bytes']:,} bytes")
        print(f"   고유 노드 수: {stats['unique_nodes']}")
    
    return 0


def cmd_benchmark(args):
    """성능 벤치마크 명령"""
    print("\n⚡ 성능 벤치마크 시작...")
    print(f"   반복 횟수: {args.iterations}")
    print(f"   샘플 수: {args.samples}")
    
    config = PipelineConfig(sample_count=args.samples)
    
    times = []
    with NoiseChainPipeline(config) as pipeline:
        for i in range(args.iterations):
            # 테스트 데이터 생성
            sensor_data = {
                "cpu_temp": 50 + 10 * np.sin(np.linspace(0, 2*np.pi, args.samples)),
                "entropy": np.random.randint(0, 256, args.samples).astype(float),
                "jitter": np.random.randn(args.samples) * 100,
            }
            
            start = time.perf_counter()
            result = pipeline.generate_and_store(sensor_data)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            
            if not result.success:
                print(f"❌ 반복 {i+1} 실패")
                return 1
            
            print(f"   [{i+1}/{args.iterations}] {elapsed*1000:.1f}ms", end="\r")
    
    print("\n")
    
    # 통계 계산
    times_ms = [t * 1000 for t in times]
    avg = np.mean(times_ms)
    std = np.std(times_ms)
    min_t = np.min(times_ms)
    max_t = np.max(times_ms)
    throughput = 1000/avg if avg > 0 else 0
    
    # 결과 요약
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"iterations": args.iterations, "samples": args.samples},
        "stats": {
            "mean_ms": float(avg),
            "std_ms": float(std),
            "min_ms": float(min_t),
            "max_ms": float(max_t),
            "throughput_tps": float(throughput)
        }
    }
    
    print(f"📊 벤치마크 결과:")
    print(f"   평균: {avg:.2f}ms")
    print(f"   표준편차: {std:.2f}ms")
    print(f"   최소: {min_t:.2f}ms")
    print(f"   최대: {max_t:.2f}ms")
    print(f"   처리량: {throughput:.1f} tokens/sec")
    
    # 파일 저장
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 결과 저장 완료: {args.output}")
        
    return 0


def cmd_demo(args):
    """전체 데모 명령"""
    print_banner()
    
    print("🎬 NoiseChain E2E 데모 시작\n")
    
    with create_pipeline() as pipeline:
        # 1. 센서 데이터 수집
        print("1️⃣  센서 데이터 수집...")
        sensor_data = {
            "cpu_temp": 50 + 10 * np.sin(np.linspace(0, 4*np.pi, 256)),
            "os_entropy": np.random.randint(0, 256, 256).astype(float),
            "clock_jitter": np.random.randn(256) * 100,
            "synthetic": np.random.randn(256) * 0.5,
        }
        print(f"   ✅ 4개 센서, 256 샘플 수집 완료\n")
        
        # 2. 토큰 생성 및 서명
        print("2️⃣  PoXToken 생성 중...")
        start = time.perf_counter()
        result = pipeline.generate_and_store(sensor_data)
        elapsed = time.perf_counter() - start
        
        token = result.token
        print(f"   ✅ 토큰 생성 완료 ({elapsed*1000:.1f}ms)")
        print(f"   📦 크기: {token.size} bytes")
        print(f"   🔐 서명: {token.signature.hex()[:32]}...\n")
        
        # 3. 검증
        print("3️⃣  토큰 검증 중...")
        report = result.verification
        print(f"   ✅ 검증 완료")
        print(f"   📋 결과: {'유효' if report.is_valid else '무효'}")
        for step in report.steps:
            icon = "✅" if step.passed else "❌"
            print(f"      {icon} {step.name}: {step.message}\n")
        
        # 4. 저장소 통계
        print("4️⃣  저장소 통계:")
        stats = pipeline.get_stats()
        print(f"   📊 총 토큰: {stats['total_tokens']}")
        print(f"   💾 총 크기: {stats['total_size_bytes']} bytes\n")
        
        print("🎉 데모 완료!")
    
    return 0


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="NoiseChain CLI - Physical Trust Verification Network"
    )
    parser.add_argument(
        "--db", 
        help="SQLite 데이터베이스 경로 (기본: 인메모리)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령")
    
    # generate 명령
    gen_parser = subparsers.add_parser("generate", help="토큰 생성")
    gen_parser.add_argument(
        "--samples", 
        type=int, 
        default=256, 
        help="수집할 샘플 수"
    )
    
    # verify 명령
    verify_parser = subparsers.add_parser("verify", help="토큰 검증")
    verify_parser.add_argument("hash", help="검증할 토큰 해시 (16진수)")
    
    # stats 명령
    subparsers.add_parser("stats", help="저장소 통계")
    
    # benchmark 명령
    bench_parser = subparsers.add_parser("benchmark", help="성능 벤치마크")
    bench_parser.add_argument(
        "--iterations", 
        type=int, 
        default=10, 
        help="반복 횟수"
    )
    bench_parser.add_argument(
        "--samples", 
        type=int, 
        default=256, 
        help="샘플 수"
    )
    bench_parser.add_argument(
        "--output", 
        help="결과 저장 파일 경로 (JSON)"
    )
    
    # demo 명령
    subparsers.add_parser("demo", help="전체 데모 실행")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "demo":
        return cmd_demo(args)
    else:
        print_banner()
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
