# NoiseChain MVP

**물리적 경험증명(PoX) 기반 신뢰 검증 네트워크**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-258%20passed-brightgreen.svg)](#테스트)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![NoiseChain Infographic](docs/noisechain_infographic.png)

## 개요

NoiseChain은 물리적 환경 노이즈(온도, 진동, EMI, 전력 변동)의 시간적 상관 구조를 서명으로 변환하여,
**"특정 장비가 특정 시간·환경을 실제로 경험했다"**를 증명하는 Physical Trust Verification Network입니다.

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 모드 설치
pip install -e .

# 테스트 실행
pytest

# 데모 실행
python -m noisechain.demo demo
```

## 사용법

### Python API

```python
from noisechain import NoiseChainPipeline

# 파이프라인 생성
with NoiseChainPipeline() as pipeline:
    # 센서 데이터 → 토큰 생성 → 서명 → 저장 → 검증
    result = pipeline.generate_and_store()
    
    print(f"Success: {result.success}")
    print(f"Token Hash: {result.token.compute_hash().hex()[:32]}...")
    print(f"Valid: {result.verification.is_valid}")
```

### CLI 데모

```bash
# 전체 데모
python -m noisechain.demo demo

# 토큰 생성
python -m noisechain.demo generate --samples 256

# 성능 벤치마크
python -m noisechain.demo benchmark --iterations 10
```

## 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  SensorHub  │───▶│TokenBuilder │───▶│TokenSigner  │
│ (4 sensors) │    │ (features)  │    │ (Ed25519)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Verification │◀───│ Repository  │◀───│  PoXToken   │
│   Engine    │    │  (SQLite)   │    │  (199 B)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 프로젝트 구조

```
NoiseChain/
├── src/noisechain/
│   ├── models/          # 데이터 모델 (Sample, TimeSeries)
│   ├── sensors/         # 가상 센서 드라이버 (4종)
│   ├── time/            # NTP 시간 동기화
│   ├── pipeline/        # 특징 추출 & 상관 서명
│   ├── token/           # PoXToken 스키마 (199 bytes)
│   ├── crypto/          # Ed25519 키 관리 & 서명
│   ├── storage/         # SQLite 토큰 저장소
│   ├── verification/    # 4단계 검증 엔진
│   ├── engine.py        # E2E 파이프라인
│   └── demo.py          # CLI 데모
├── tests/               # 258개 테스트 케이스
├── docs/                # 설계 문서
├── pyproject.toml       # 프로젝트 설정
└── requirements.txt     # 의존성
```

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **PoX Token** | 물리적 경험 증명 토큰 (199 bytes) |
| **Noise Fingerprint** | 복제 불가능한 노이즈 지문 (99 bytes) |
| **Correlation Signature** | 다중 센서 상관 구조 서명 (SHA3-256) |
| **Risk Score** | 0-100% 위험 점수 기반 판정 |

## 검증 파이프라인

```
1. Schema Validation    ─▶ 필드 크기, 범위 검증
2. Signature Verify     ─▶ Ed25519 서명 검증
3. Timestamp Check      ─▶ 미래/수명 초과 검사
4. Risk Score Assess    ─▶ 임계값 비교
```

## 테스트

```bash
# 전체 테스트
pytest

# 커버리지 리포트
pytest --cov=noisechain --cov-report=html
```

| 모듈 | 테스트 수 |
|------|----------|
| models | 28 |
| sensors | 42 |
| time | 25 |
| features | 29 |
| correlation | 21 |
| token | 24 |
| crypto | 24 |
| storage | 21 |
| verification | 25 |
| pipeline | 18 |
| **Total** | **258** |

## 성능

- **토큰 생성**: ~10ms (256 샘플)
- **토큰 크기**: 199 bytes (서명 포함)
- **처리량**: ~100 tokens/sec

## 의존성

- Python 3.11+
- numpy
- scipy
- pynacl (Ed25519)
- ntplib

## 라이선스

MIT License

## 저자

**Jung Wook Yang**  
📧 <sadpig70@gmail.com>  
🔗 [GitHub](https://github.com/sadpig70/NoiseChain)
