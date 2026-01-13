# NoiseChain MVP 상세 설계서

**문서 버전**: 1.0  
**작성일**: 2026-01-13  
**상태**: 설계중 (Design Phase)  
**범위**: Phase 0 시뮬레이션 MVP (0-3개월)  
**개발 환경**: 시뮬레이션 기반 (하드웨어 센서 없이 가상 노이즈 사용)

---

## 1. MVP 목표

- 가상 센서 기반 PoXToken 생성 및 검증 파이프라인 완성
- 리플레이 공격 탐지 및 위조 탐지 알고리즘 검증
- 데모 시나리오를 통한 핵심 가치 시연

---

## 2. MVP 전체 Gantree

```
NoiseChain_MVP // Phase 0 시뮬레이션 MVP (설계중)
    EdgeLayer_MVP // 가상 센서 기반 노이즈 수집 (설계중)
        VirtualSensorDriver // 가상 센서 드라이버 (설계중)
        NoiseDataModel // 노이즈 데이터 모델 (설계중)
        RingBuffer // 순환 버퍼 (설계중)
    TimeSyncModule_MVP // NTP 시간 동기화 (설계중)
        NTPClient // NTP 클라이언트 (설계중)
        TimestampGenerator // 타임스탬프 생성 (설계중)
    AttestationPipeline_MVP // PoX 생성 핵심 (설계중)
        FeatureExtractor_MVP // 특징 추출 (설계중)
        CorrelationSignature // 상관 서명 알고리즘 (설계중)
        PoXToken_v1 // 토큰 스키마 (설계중)
        TokenAssembler // 토큰 조립/서명 (설계중)
    LedgerStorage_MVP // SQLite 저장소 (설계중)
        SQLiteAdapter // SQLite 연동 (설계중)
        TokenStore // 토큰 저장 (설계중)
        IndexStore // 인덱스 관리 (설계중)
    VerificationLayer_MVP // 검증 엔진 (설계중)
        SchemaValidator // 스키마 검증 (설계중)
        SignatureVerifier // 서명 검증 (설계중)
        CorrelationMatcher // 상관 패턴 매칭 (설계중)
        RiskScoreCalculator // 위험점수 계산 (설계중)
        DecisionEngine // 판정 엔진 (설계중)
    DemoScenarios // 데모 시나리오 (설계중)
        ReplayAttackDemo // 리플레이 공격 시연 (설계중)
        TamperDetectionDemo // 위조 탐지 시연 (설계중)
        TokenLifecycleDemo // 전체 생명주기 시연 (설계중)
```

---

## 3. 가상 센서 데이터 모델 (NoiseDataModel)

### 3.1 Gantree 상세

```
NoiseDataModel // 가상 센서 데이터 모델 (설계중)
    SensorType // 센서 타입 열거 (설계중)
        CPUTemp // CPU 온도 센서 (설계중)
        OSEntropy // OS 엔트로피 (설계중)
        ClockJitter // 클럭 지터 (설계중)
        SyntheticNoise // 합성 노이즈 (설계중)
    Sample // 단일 샘플 (설계중)
        timestamp_ns // 나노초 타임스탬프 (설계중)
        sensor_values // 센서별 값 맵 (설계중)
    TimeSeries // 시계열 버퍼 (설계중)
        samples // 샘플 벡터 (설계중)
        sampling_rate_hz // 샘플링 레이트 (설계중)
        window_size // 윈도우 크기 (설계중)
```

### 3.2 데이터 구조 정의

```rust
// 센서 타입 열거
enum SensorType {
    CPUTemp,       // 범위: 30.0 ~ 100.0 (°C)
    OSEntropy,     // 범위: 0 ~ 255 (byte)
    ClockJitter,   // 범위: -1000 ~ 1000 (ns)
    SyntheticNoise // 범위: -1.0 ~ 1.0 (normalized)
}

// 단일 샘플
struct Sample {
    timestamp_ns: i64,                    // 나노초 타임스탬프
    sensor_values: HashMap<SensorType, f64>  // 센서별 측정값
}

// 시계열 버퍼
struct TimeSeries {
    samples: Vec<Sample>,      // 샘플 배열
    sampling_rate_hz: u32,     // 샘플링 레이트 (기본: 100 Hz)
    window_size: u32           // 윈도우 크기 (기본: 256 샘플)
}
```

### 3.3 가상 센서 구현

| 센서 타입 | 데이터 소스 | 구현 방법 |
|----------|------------|----------|
| CPUTemp | `psutil.sensors_temperatures()` | 시스템 API 호출 |
| OSEntropy | `/dev/urandom` 또는 `os.urandom()` | 바이트 읽기 |
| ClockJitter | `time.perf_counter_ns()` 연속 호출 차이 | 타이밍 측정 |
| SyntheticNoise | `numpy.random.normal(0, 0.3)` | 가우시안 노이즈 |

---

## 4. PoXToken v1 스키마

### 4.1 Gantree 상세

```
PoXToken_v1 // PoX 토큰 스키마 v1 (설계중)
    Header // 토큰 헤더 (설계중)
        token_id // 토큰 고유 ID (bytes[32])
        version // 스키마 버전 (u8)
        device_id // 장비 식별자 (string[64])
        profile_id // 프로파일 ID (string[32])
        created_at // 생성 시간 (i64, ns)
    TimeWindow // 시간창 정보 (설계중)
        t_start // 시작 타임스탬프 (i64, ns)
        t_end // 종료 타임스탬프 (i64, ns)
        window_size_ms // 윈도우 크기 (u32, ms)
    NoiseSignature // 노이즈 서명 (설계중)
        feature_vector // 특징 벡터 ([f64; 16])
        correlation_hash // 상관 해시 (bytes[32])
        sensor_count // 센서 개수 (u8)
    Challenge // 리플레이 방지 (설계중)
        nonce // 검증자 nonce (bytes[16])
        challenge_id // 챌린지 세션 ID (string[32])
    Integrity // 무결성 보장 (설계중)
        prev_hash // 이전 토큰 해시 (bytes[32])
        merkle_root // 머클 루트 (bytes[32], optional)
        signature // Ed25519 서명 (bytes[64])
```

### 4.2 직렬화 형식

```rust
// PoXToken v1 구조체 (총 약 430 bytes)
struct PoXToken_v1 {
    // Header (약 145 bytes)
    token_id: [u8; 32],       // 32 bytes - SHA3-256(내용)
    version: u8,              // 1 byte - 0x01
    device_id: String,        // max 64 bytes
    profile_id: String,       // max 32 bytes
    created_at: i64,          // 8 bytes - nanoseconds since epoch
    
    // TimeWindow (20 bytes)
    t_start: i64,             // 8 bytes - window start (ns)
    t_end: i64,               // 8 bytes - window end (ns)
    window_size_ms: u32,      // 4 bytes - window size (ms)
    
    // NoiseSignature (161 bytes)
    feature_vector: [f64; 16], // 128 bytes (16 * 8)
    correlation_hash: [u8; 32], // 32 bytes - SHA3-256
    sensor_count: u8,          // 1 byte
    
    // Challenge (50 bytes)
    nonce: [u8; 16],          // 16 bytes - random bytes
    challenge_id: String,     // max 32 bytes
    
    // Integrity (96~128 bytes)
    prev_hash: [u8; 32],      // 32 bytes - previous token hash
    merkle_root: Option<[u8; 32]>, // 0 or 32 bytes - optional
    signature: [u8; 64]       // 64 bytes - Ed25519 signature
}
```

### 4.3 JSON 표현 (디버깅/API용)

```json
{
  "header": {
    "token_id": "0x8f3e7a...",
    "version": 1,
    "device_id": "sim_device_001",
    "profile_id": "simulation_v1",
    "created_at": 1705132991123456789
  },
  "time_window": {
    "t_start": 1705132990000000000,
    "t_end": 1705132991000000000,
    "window_size_ms": 1000
  },
  "noise_signature": {
    "feature_vector": [0.23, -0.15, 0.87, ...],
    "correlation_hash": "0x9c7f2b...",
    "sensor_count": 4
  },
  "challenge": {
    "nonce": "0x7d1e3c...",
    "challenge_id": "sess_abc123"
  },
  "integrity": {
    "prev_hash": "0x4a2c8f...",
    "merkle_root": null,
    "signature": "0x3b8a1d..."
  }
}
```

---

## 5. Correlation Signature 알고리즘

### 5.1 Gantree 상세

```
CorrelationSignature // 상관 서명 알고리즘 (설계중)
    Input // 입력 정의 (설계중)
        multi_sensor_data // 다중 센서 시계열
        window_size // 샘플 수
    LagComputation // 라그 계산 (설계중)
        max_lag // 최대 라그 (±10 샘플)
        lag_step // 라그 간격 (1 샘플)
        lag_range // 라그 범위 생성
    CorrelationMatrix // 상관 행렬 (설계중)
        cross_correlation // 교차 상관 계산
        pearson_coefficient // 피어슨 상관 계수
        matrix_build // 행렬 구성
    Quantization // 양자화 (설계중)
        normalize // 정규화 (MinMax)
        quantize_bits // 비트 양자화 (8비트)
        flatten // 벡터화
    Hashing // 해시 생성 (설계중)
        concatenate // 특징 연결
        hash_sha3_256 // SHA3-256 해시
    Verification // 검증 (설계중)
        similarity_metric // 유사도 메트릭
        threshold_check // 임계값 검사
```

### 5.2 알고리즘 정의

```python
# Correlation Signature 알고리즘 (Python 프로토타입)

import numpy as np
from scipy import signal
import hashlib

class CorrelationSignature:
    # 파라미터
    MAX_LAG = 10          # 최대 라그 (샘플)
    LAG_STEP = 1          # 라그 간격
    QUANT_BITS = 8        # 양자화 비트
    SIMILARITY_THRESHOLD = 0.85  # 유사도 임계값
    
    def compute(self, sensor_data: dict[str, np.ndarray]) -> tuple[np.ndarray, bytes]:
        """
        다중 센서 데이터로부터 상관 서명 생성
        
        Args:
            sensor_data: {sensor_name: time_series} 딕셔너리
        
        Returns:
            (feature_vector, correlation_hash)
        """
        sensors = list(sensor_data.keys())
        n_sensors = len(sensors)
        lags = range(-self.MAX_LAG, self.MAX_LAG + 1, self.LAG_STEP)
        n_lags = len(lags)
        
        # 1. 교차 상관 행렬 계산 (N x N x L)
        corr_matrix = np.zeros((n_sensors, n_sensors, n_lags))
        for i, s1 in enumerate(sensors):
            for j, s2 in enumerate(sensors):
                if i <= j:
                    corr = signal.correlate(
                        sensor_data[s1], 
                        sensor_data[s2], 
                        mode='same'
                    )
                    # 중앙 ±MAX_LAG 추출
                    center = len(corr) // 2
                    corr_matrix[i, j, :] = corr[center - self.MAX_LAG : center + self.MAX_LAG + 1]
                    corr_matrix[j, i, :] = corr_matrix[i, j, :]  # 대칭
        
        # 2. 정규화 (MinMax)
        corr_min = corr_matrix.min()
        corr_max = corr_matrix.max()
        normalized = (corr_matrix - corr_min) / (corr_max - corr_min + 1e-10)
        
        # 3. 특징 벡터 추출 (상삼각 + 대각선)
        feature_list = []
        for i in range(n_sensors):
            for j in range(i, n_sensors):
                # 각 센서 쌍의 최대 상관 라그와 값
                max_idx = np.argmax(np.abs(normalized[i, j, :]))
                feature_list.append(normalized[i, j, max_idx])
        
        feature_vector = np.array(feature_list[:16])  # 최대 16개
        
        # 4. 양자화 (8비트)
        quantized = np.round(feature_vector * (2**self.QUANT_BITS - 1)).astype(np.uint8)
        
        # 5. 해시 생성
        hash_input = quantized.tobytes() + normalized.tobytes()
        correlation_hash = hashlib.sha3_256(hash_input).digest()
        
        return feature_vector, correlation_hash
    
    def verify(self, 
               original_hash: bytes, 
               original_features: np.ndarray,
               new_hash: bytes,
               new_features: np.ndarray) -> tuple[bool, float]:
        """
        상관 서명 검증
        
        Returns:
            (is_valid, similarity_score)
        """
        # 해시 일치 확인
        hash_match = (original_hash == new_hash)
        
        # 특징 벡터 유사도 (코사인 유사도)
        dot = np.dot(original_features, new_features)
        norm_orig = np.linalg.norm(original_features)
        norm_new = np.linalg.norm(new_features)
        similarity = dot / (norm_orig * norm_new + 1e-10)
        
        is_valid = hash_match or (similarity >= self.SIMILARITY_THRESHOLD)
        
        return is_valid, similarity
```

> **주의**: `verify()` 메서드는 새 센서 데이터의 해시와 특징을 전달받아 비교합니다.
> 검증 시 저장된 원본 토큰과 새로 제출된 토큰을 비교하는 것이 아니라,
> 저장된 토큰의 해시/특징과 챌린지 응답의 해시/특징을 비교합니다.

### 5.3 파라미터 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `MAX_LAG` | 10 | 최대 시간 지연 (샘플 수) |
| `LAG_STEP` | 1 | 라그 간격 |
| `QUANT_BITS` | 8 | 양자화 비트 수 |
| `SIMILARITY_THRESHOLD` | 0.85 | 검증 시 최소 유사도 |
| `FEATURE_SIZE` | 16 | 특징 벡터 크기 |

---

## 6. RiskScore 계산 로직

### 6.1 Gantree 상세

```
RiskScoreCalculator // 위험점수 계산기 (설계중)
    Components // 점수 구성요소 (설계중)
        time_validity_score // 시간 유효성 점수 (설계중)
        signature_match_score // 서명 일치 점수 (설계중)
        correlation_similarity_score // 상관 유사도 점수 (설계중)
        chain_integrity_score // 체인 무결성 점수 (설계중)
    Weights // 가중치 설정 (설계중)
        weight_config // 가중치 구성 (설계중)
    Aggregation // 점수 집계 (설계중)
        weighted_sum // 가중 합계 (설계중)
        normalize // 정규화 (설계중)
    Decision // 판정 (설계중)
        thresholds // 임계값 정의 (설계중)
        decision_logic // 판정 로직 (설계중)
```

### 6.2 공식 정의

```python
class RiskScoreCalculator:
    # 가중치 (합계 = 1.0)
    WEIGHTS = {
        'time_validity': 0.20,      # 시간 유효성
        'signature_match': 0.25,    # 서명 일치
        'correlation_sim': 0.40,    # 상관 유사도 (가장 중요)
        'chain_integrity': 0.15     # 체인 무결성
    }
    
    # 판정 임계값
    THRESHOLDS = {
        'accept_max': 0.30,   # risk < 0.30 → accept
        'review_max': 0.70,   # 0.30 ≤ risk < 0.70 → review
        # risk ≥ 0.70 → reject
    }
    
    def calculate(self, 
                  time_valid: bool,
                  sig_match: bool,
                  corr_similarity: float,
                  chain_intact: bool) -> tuple[float, str]:
        """
        위험점수 계산 및 판정
        
        Args:
            time_valid: 시간창 유효성 (True/False)
            sig_match: 서명 일치 (True/False)
            corr_similarity: 상관 유사도 (0.0 ~ 1.0)
            chain_intact: 체인 무결성 (True/False)
        
        Returns:
            (risk_score, decision)
        """
        # 개별 점수 (높을수록 위험)
        scores = {
            'time_validity': 0.0 if time_valid else 1.0,
            'signature_match': 0.0 if sig_match else 1.0,
            'correlation_sim': 1.0 - corr_similarity,  # 유사도 반전
            'chain_integrity': 0.0 if chain_intact else 1.0
        }
        
        # 가중 합계
        risk_score = sum(
            scores[k] * self.WEIGHTS[k] 
            for k in self.WEIGHTS
        )
        
        # 판정
        if risk_score < self.THRESHOLDS['accept_max']:
            decision = 'accept'
        elif risk_score < self.THRESHOLDS['review_max']:
            decision = 'review'
        else:
            decision = 'reject'
        
        return risk_score, decision
```

### 6.3 판정 매트릭스

| 조건 | risk_score 범위 | 판정 | 조치 |
|------|----------------|------|------|
| 모든 검증 통과 | 0.00 ~ 0.29 | **accept** | 즉시 승인 |
| 일부 이상 징후 | 0.30 ~ 0.69 | **review** | 수동 검토 |
| 명확한 위조 | 0.70 ~ 1.00 | **reject** | 거부 + 알림 |

---

## 7. 데모 시나리오

### 7.1 Gantree 상세

```
DemoScenarios // 데모 시나리오 (설계중)
    ReplayAttackDemo // 리플레이 공격 탐지 시연 (설계중)
        step1_generate_valid_token // 정상 토큰 생성
        step2_save_token // 토큰 저장
        step3_wait_time_window // 시간창 경과 대기
        step4_replay_token // 토큰 재전송 시도
        step5_detect_replay // 리플레이 탐지 확인
        expected_result // 예상 결과: reject
    TamperDetectionDemo // 위조 탐지 시연 (설계중)
        step1_generate_token // 토큰 생성
        step2_tamper_feature // 특징 벡터 변조
        step3_submit_tampered // 변조 토큰 제출
        step4_verify // 검증 실행
        expected_result // 예상 결과: reject
    TokenLifecycleDemo // 전체 생명주기 시연 (설계중)
        step1_collect_noise // 노이즈 수집
        step2_extract_features // 특징 추출
        step3_generate_signature // 서명 생성
        step4_mint_token // 토큰 발행
        step5_store_ledger // 원장 저장
        step6_query_token // 토큰 조회
        step7_verify_token // 토큰 검증
        expected_result // 예상 결과: accept
```

### 7.2 시나리오 상세

#### Demo 1: 리플레이 공격 탐지

```
시퀀스:
1. 가상 센서에서 노이즈 수집 → PoXToken 생성
2. 토큰 저장 (시간: T0)
3. 시간창 1개 경과 대기 (T0 + window_size)
4. 동일 토큰으로 검증 재요청
5. 검증 결과: reject (nonce 만료 + 시간창 초과)

검증 포인트:
- nonce 재사용 탐지
- 시간창 유효성 검사
- 해시체인 연속성 검사
```

#### Demo 2: 위조 탐지

```
시퀀스:
1. 정상 PoXToken 생성
2. feature_vector[0]을 +0.5 변조
3. 변조된 토큰으로 검증 요청
4. 검증 결과: reject (correlation_hash 불일치)

검증 포인트:
- 특징 벡터 무결성
- correlation_hash 검증
- signature 불일치 탐지
```

---

## 8. 원자화 노드 구현 순서

### 8.1 구현 우선순위

| 순서 | 노드 | 복잡도 | 예상 시간 | 의존성 |
|-----|------|--------|----------|--------|
| 1 | NoiseDataModel | 낮음 | 30분 | 없음 |
| 2 | VirtualSensorDriver | 중간 | 1시간 | NoiseDataModel |
| 3 | TimestampGenerator | 낮음 | 20분 | 없음 |
| 4 | FeatureExtractor_MVP | 중간 | 1시간 | NoiseDataModel |
| 5 | CorrelationSignature | 높음 | 2시간 | FeatureExtractor |
| 6 | PoXToken_v1 | 중간 | 1시간 | CorrelationSignature |
| 7 | TokenAssembler | 중간 | 1시간 | PoXToken_v1 |
| 8 | SQLiteAdapter | 중간 | 1시간 | PoXToken_v1 |
| 9 | SignatureVerifier | 중간 | 1시간 | PoXToken_v1 |
| 10 | CorrelationMatcher | 높음 | 1.5시간 | CorrelationSignature |
| 11 | RiskScoreCalculator | 중간 | 45분 | Matcher + Verifier |
| 12 | DecisionEngine | 낮음 | 30분 | RiskScoreCalculator |
| 13 | DemoScenarios | 낮음 | 1시간 | 전체 |

**총 예상 시간**: 약 12시간 (2~3일)

---

## 9. 기술 스택 (MVP)

| 영역 | 기술 | 선택 이유 |
|------|------|----------|
| Language | Python 3.11+ | 빠른 프로토타이핑 |
| Crypto | `hashlib`, `ed25519` | 표준 라이브러리 |
| Signal | `numpy`, `scipy` | 상관 계산 |
| Storage | `sqlite3` | 내장 DB |
| Time | `time.perf_counter_ns()` | 나노초 정밀도 |
| Sensor | `psutil` | 시스템 정보 |

---

## 10. 검증 체크리스트

### 기능 검증

- [ ] 가상 센서에서 노이즈 수집 가능
- [ ] 특징 추출 알고리즘 동작
- [ ] 상관 서명 생성/검증 동작
- [ ] PoXToken 직렬화/역직렬화
- [ ] SQLite 저장/조회 동작
- [ ] RiskScore 계산 정확성
- [ ] 리플레이 공격 탐지 성공
- [ ] 위조 탐지 성공

### 성능 목표

- [ ] 토큰 생성 시간 < 100ms
- [ ] 토큰 검증 시간 < 50ms
- [ ] 저장소 조회 시간 < 10ms

---

*— End of MVP Design Document —*
