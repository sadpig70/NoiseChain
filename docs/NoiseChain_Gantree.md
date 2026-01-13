# NoiseChain 공식 Gantree 설계서

**문서 버전**: 1.0  
**작성일**: 2026-01-13  
**상태**: 설계중 (Design Phase)  
**개발 환경**: 시뮬레이션 기반 (하드웨어 센서 없이 가상 노이즈 사용)  
**통합 출처**: ChatGPT, Gemini, Claude, Grok, Kimi, DeepSeek, Qwen, Perplexity

---

## 1. 시스템 루트 트리 (Level 0-2)

```
NoiseChain_System // 물리적 경험증명(PoX) 기반 신뢰 검증 네트워크 (설계중)
    EdgeLayer // 현장 센서/노이즈 수집 계층 (설계중)
    TimeSyncModule // 시간 동기화 모듈 (설계중)
    AttestationPipeline // PoX 생성 파이프라인 (설계중)
    VerificationLayer // 검증 서비스 계층 (설계중)
    LedgerStorage // 불변 저장/감사 로그 (설계중)
    ProfileRegistry // 업종별 증명 레시피 (설계중)
    NetworkLayer // P2P 네트워크 계층 (설계중)
    APILayer // 고객 대면 인터페이스 (설계중)
    SecurityFramework // 보안/위협 모델 (설계중)
    OpsAndObservability // 운영/관측성 (설계중)
```

---

## 2. EdgeLayer 분해 트리

```
EdgeLayer // 현장 센서/노이즈 수집 계층 (설계중)
    SensorInterface // 다중 센서 인터페이스 (설계중)
        VirtualSensorDriver // [MVP] 가상 센서 드라이버 (설계중)
            CPUTempReader // CPU 온도 읽기 (psutil) (설계중)
            OSEntropyReader // OS 엔트로피 (/dev/urandom) (설계중)
            SyntheticNoiseGen // numpy 기반 합성 노이즈 (설계중)
            ClockJitterReader // 타이밍 변동 측정 (설계중)
        TempSensorDriver // 온도 센서 드라이버 (보류-하드웨어)
        VibrationSensorDriver // 진동 센서 드라이버 (보류-하드웨어)
        EMISensorDriver // 전자기 간섭 센서 (보류-하드웨어)
        PowerNoiseSensor // 전력 노이즈 센서 (보류-하드웨어)
        ClockJitterSensor // 클럭 지터 센서 (보류-하드웨어)
    NoiseCapture // 노이즈 캐처 엔진 (설계중)
        SamplingScheduler // 샘플링 주기 제어 (설계중)
            SampleRateCalculator // 샘플링 레이트 계산 (완료)
            AdaptiveInterval // 적응형 샘플링 간격 (설계중)
        RingBuffer // 순환 버퍼 관리 (설계중)
        HealthMonitor // 센서 고장/이상치 탐지 (설계중)
    DataNormalizer // 데이터 정규화 (설계중)
        ScaleNormalizer // 스케일 정규화 (설계중)
        FrequencyNormalizer // 주파수 정규화 (설계중)
        OutlierRemoval // 이상치 제거 (설계중)
    SecureIdentity // 장비 식별/키 보관 (설계중)
        DeviceRootKey // 디바이스 루트 키 (HSM/TPM) (보류)
        KeyRotation // 주기적 키 로테이션 (보류)
```

---

## 3. TimeSyncModule 분해 트리

```
TimeSyncModule // 시간 동기화 모듈 (설계중)
    NTPClient // NTP 시간 동기화 (설계중)
        NTPServerPool // NTP 서버 풀 관리 (설계중)
        ClockSkewDetector // 시계 편차 감지 (설계중)
    PTPClient // IEEE 1588 정밀 시간 (설계중)
        PTPMasterSelector // 마스터 클럭 선택 (설계중)
        LeapSecondHandler // 윤초 처리 (설계중)
    GPSTimeReference // GPS PPS 기준 시간 (보류)
        SignalIntegrityChecker // GPS 스푸핑 탐지 (보류)
        HoldoverController // 신호 손실 시 유지 (보류)
    TimestampGenerator // 나노초 타임스탬프 생성 (설계중)
    DriftEstimator // 드리프트 추정/보정 (설계중)
```

---

## 4. AttestationPipeline 분해 트리 (핵심)

```
AttestationPipeline // PoX 생성 파이프라인 (설계중)
    WindowAcquisition // 시간창 다중센서 동시 수집 (설계중)
        TimeWindowSlicer // 시간창 분할기 (설계중)
        AdaptiveWindowSizer // 적응형 창 크기 조정 (설계중)
    Preprocessor // 전처리/정규화/결측 처리 (설계중)
    FeatureExtractor // 특징 추출 (분해)
    CorrelationBuilder // 상관 서명 생성 (분해)
    ChallengeResponse // 리플레이/스푸핑 방어 (설계중)
        NonceFromVerifier // 검증자 nonce 포함 (설계중)
        AntiReplayGuard // 리플레이 공격 방지 (설계중)
    TokenAssembler // 토큰 조립/서명 (설계중)
        TokenHashChain // 연속 윈도우 해시 체인 (설계중)
        DeviceSignature // 디바이스 키 서명 (설계중)
        GatewayCoSign // 게이트웨이 공동서명 (보류)
```

### FeatureExtractor 분해

```
FeatureExtractor // 특징 추출기 (설계중)
    StatFeatures // 통계적 특징 (설계중)
        MeanVariance // 평균/분산 계산 (완료)
        Skewness // 왜도 계산 (설계중)
        Kurtosis // 첨도 계산 (설계중)
        Entropy // 엔트로피 계산 (설계중)
            ShannonEntropy // 섀넌 엔트로피 (완료)
            PermutationEntropy // 순열 엔트로피 (설계중)
    FreqFeatures // 주파수 특징 (설계중)
        FFTAnalyzer // FFT 분석기 (설계중)
        PeakDetector // 피크 주파수 검출 (설계중)
        BandPower // 주파수 대역 파워 (설계중)
    TemporalFeatures // 시간 특징 (설계중)
        AutoCorrelation // 자기상관 계산 (설계중)
        CrossCorrelation // 교차상관 계산 (설계중)
        TrendDetector // 추세 검출 (설계중)
    FeatureVector // 특징 벡터 조합 (설계중)
        VectorConcatenator // 벡터 결합 (설계중)
        VectorNormalizer // 최종 정규화 (설계중)
```

### CorrelationBuilder 분해

```
CorrelationBuilder // 상관 서명 생성 (설계중)
    LaggedCorrelationMatrix // 라그별 상관행렬 (설계중)
        MultiLagComputer // 다중 라그 계산 (설계중)
        OTOCMatrix // OTOC 행렬 구성 (완료)
    QuantizeAndSketch // 양자화→스케치→해시 (설계중)
        Quantizer // 양자화 모듈 (설계중)
        SketchCompressor // 스케치 압축 (설계중)
    RobustnessGuard // 온도/노후화 강인성 (설계중)
```

---

## 5. VerificationLayer 분해 트리

```
VerificationLayer // 검증 서비스 계층 (설계중)
    VerifyAPI // 토큰 검증 API (설계중)
        SchemaValidator // 스키마/버전 검증 (설계중)
        TimeWindowValidator // 시간창/nonce 검증 (설계중)
        ConsistencyChecker // 해시체인/서명 검증 (설계중)
    MatchingAndScoring // 위험점수 중심 판정 (설계중)
        ProfileThresholds // 프로파일별 임계값 (설계중)
        RiskScoreCalculator // 0~1 위험점수 계산 (설계중)
        DecisionEngine // accept/review/reject (설계중)
    NoisePatternMatcher // 노이즈 패턴 매칭 (설계중)
        SimilarityCalculator // 유사도 계산 (설계중)
        FuzzyMatcher // 허용 오차 유사도 비교 (설계중)
        DistanceMetric // 거리 메트릭 (코사인) (설계중)
    EvidenceReceipt // 검증 영수증 발급 (설계중)
        ReceiptBuilder // 영수증 생성 (설계중)
        ReceiptSigner // 영수증 서명 (설계중)
```

---

## 6. LedgerStorage 분해 트리

```
LedgerStorage // 불변 저장/감사 로그 (설계중)
    AppendOnlyLog // 불변 로그 저장소 (설계중)
        ObjectStore // PoXToken/Receipt 저장 (설계중)
        IndexStore // device_id/profile_id/time 인덱싱 (설계중)
    StorageBackend // 저장소 백엔드 (설계중)
        SQLiteAdapter // SQLite 어댑터 (설계중)
        TimescaleDB // 시계열 DB (설계중)
        DistributedStore // 분산 저장소 (보류)
    MerkleAnchoring // 무결성 앵커 (설계중)
        DailyMerkleRoot // 일 단위 머클루트 (설계중)
        MerkleTreeBuilder // 머클 트리 구성 (설계중)
        ExternalAnchor // 퍼블릭 체인 앵커 (보류)
    AuditQuery // 감사/리포트 조회 (설계중)
        TraceQuery // 이력 조회 (설계중)
        ComplianceExport // 규제 제출 포맷 (보류)
    HashChain // SHA-256 체인 구조 (설계중)
        BlockCreation // 블록 생성 (완료)
        ChainValidation // 체인 검증 (설계중)
```

---

## 7. ProfileRegistry 분해 트리

```
ProfileRegistry // 업종별 증명 레시피 (설계중)
    ProfileVersioning // v1,v2... 버전 관리 (설계중)
    RecipeLibrary // 센서셋/특징/라그 레시피 (설계중)
        SensorSetDefinition // 필수 센서 목록 (설계중)
        FeatureRecipe // 추출 특징/라그 규칙 (설계중)
        AcceptancePolicy // FR/FA 목표, 임계값 (설계중)
    Governance // 프로파일 권한 통제 (설계중)
        ProfileAdmin // 정책 정의자 (설계중)
        SignedProfile // 서명된 프로파일 (설계중)
    IndustryProfiles // 산업별 프로파일 (설계중)
        ColdChainProfile // 콜드체인 (설계중)
        LabIntegrityProfile // 임상시험 (설계중)
        SupplyChainProfile // 공급망 (설계중)
        ESGProfile // ESG 탄소배출 (보류)
```

---

## 8. NetworkLayer 분해 트리

```
NetworkLayer // P2P 네트워크 계층 (설계중)
    NodeDiscovery // 노드 탐색 (설계중)
        BootstrapNodes // 부트스트랩 노드 (설계중)
        PeerExchange // 피어 교환 (설계중)
        HeartbeatMonitor // 하트비트 모니터 (설계중)
    P2PProtocol // P2P 프로토콜 (설계중)
        MessageCodec // 메시지 코덱 (설계중)
        ChannelManager // 채널 관리자 (설계중)
        FlowControl // 흐름 제어 (설계중)
    SyncManager // 동기화 관리자 (설계중)
        ProofSync // 증명 동기화 (설계중)
        StateSync // 상태 동기화 (설계중)
        ConflictResolver // 충돌 해결기 (설계중)
    SecurityLayer // 보안 계층 (설계중)
        TLSWrapper // TLS 래퍼 (설계중)
        NodeAuthenticator // 노드 인증기 (설계중)
        AntiSpam // 스팸 방지 (설계중)
```

---

## 9. APILayer 분해 트리

```
APILayer // 고객 대면 인터페이스 (설계중)
    RESTInterface // REST API (설계중)
        DeviceRegistration // 장치 등록 API (설계중)
        NoiseSubmission // 노이즈 제출 API (설계중)
        ProofVerification // 증명 검증 API (설계중)
        ChallengeRequest // 챌린지 요청 API (설계중)
    GraphQLInterface // GraphQL 인터페이스 (보류)
    WebhookIntegration // Webhook 연동 (설계중)
        RealTimeAlert // 실시간 경고 (설계중)
        ComplianceNotification // 규제 알림 (설계중)
    SDKs // 클라이언트 SDK (설계중)
        EdgeSDK // Edge 노드 SDK (설계중)
        GatewaySDK // Gateway SDK (설계중)
        VerifyClientSDK // 검증 클라이언트 SDK (설계중)
    WebConsole // 관리 콘솔 UI (보류)
        DeviceMonitor // 장치 모니터링 (보류)
        ProfileManager // 프로파일 관리 (보류)
        AuditReportViewer // 감사 리포트 (보류)
```

---

## 10. SecurityFramework 분해 트리

```
SecurityFramework // 보안/위협 모델 (설계중)
    ThreatModel // 위협 분석 (설계중)
        ReplayAttack // 리플레이 공격 방어 (설계중)
        SensorSpoofing // 센서 스푸핑 방어 (설계중)
        TimeManipulation // 시간 조작 방어 (설계중)
        InsiderAttack // 내부자 공격 방어 (설계중)
    DefenseMechanism // 방어 메커니즘 (설계중)
        NonceBasedAntiReplay // Nonce 기반 방어 (설계중)
        MultiSensorCorrelation // 다중 센서 상관 검증 (설계중)
        HashChainIntegrity // 해시 체인 무결성 (설계중)
    KeyManagement // 키 관리 (설계중)
        QuantumResistantCrypto // 양자 내성 암호 (설계중)
        HSMIntegration // HSM 연동 (설계중)
        KeyRotationPolicy // 키 로테이션 정책 (설계중)
    AccessControl // 접근 제어 (설계중)
        RoleBasedAuth // 역할 기반 권한 (설계중)
        AuditLogging // 감사 로깅 (설계중)
    ComplianceModule // 규제 준수 (설계중)
        GDPR_Compliance // GDPR (설계중)
        FDA_DSCSA_Compliance // FDA DSCSA (설계중)
        EU_BatteryPass // EU Battery Pass (설계중)
```

---

## 11. OpsAndObservability 분해 트리

```
OpsAndObservability // 운영/관측성 (설계중)
    Metrics // 운영 지표 (설계중)
        TokenSuccessRate // 토큰 발급 성공률 (설계중)
        VerificationLatency // 검증 지연 (설계중)
        TokenDropRate // 토큰 유실률 (설계중)
        DriftRate // 드리프트율 (설계중)
    SLOs // 서비스 수준 목표 (설계중)
        AvailabilityTarget // 가용성 목표 (설계중)
        LatencyTarget // 지연 목표 (설계중)
    IncidentPlaybooks // 장애 대응 (설계중)
        KeyCompromise // 키 유출 대응 (설계중)
        SensorFailure // 센서 이상 대응 (설계중)
        TimeSyncFailure // 시간 동기 장애 대응 (설계중)
    Monitoring // 모니터링 (설계중)
        RealTimeAnalytics // 실시간 분석 (설계중)
        AnomalyVisualization // 이상 탐지 시각화 (설계중)
```

---

## 12. 원자화 노드 목록 (구현 가능 단위)

| # | 노드명 | 영역 | 예상 시간 | 상태 |
|---|--------|------|----------|------|
| 1 | SampleRateCalculator | 신호처리 | 10분 | 완료 |
| 2 | ShannonEntropy | 통계 | 5분 | 완료 |
| 3 | MeanVariance | 통계 | 3분 | 완료 |
| 4 | OTOCMatrix | 선형대수 | 20분 | 완료 |
| 5 | BlockCreation | 암호학 | 10분 | 완료 |
| 6 | ZScoreCalculation | 통계 | 1분 | 설계중 |
| 7 | WeightedSum | 벡터연산 | 5분 | 설계중 |
| 8 | SimulationExecution | 양자시뮬 | 15분 | 설계중 |
| 9 | MerkleProof | 암호학 | 15분 | 설계중 |
| 10 | TimeSync | 시간동기 | 20분 | 설계중 |
| 11 | ReplayAttackCheck | 보안 | 10분 | 설계중 |
| 12 | APIRateLimiter | API관리 | 15분 | 설계중 |
| 13 | ChallengePuzzle | PoX생성 | 20분 | 설계중 |
| 14 | FuzzyMatcher | 검증 | 25분 | 설계중 |

---

## 13. 의존성 흐름

```
EdgeLayer (입력)
    ↓
TimeSyncModule (타임스탬프)
    ↓
AttestationPipeline
    ├─→ FeatureExtractor → CorrelationBuilder
    └─→ TokenAssembler
    ↓
VerificationLayer
    ├─→ MatchingAndScoring
    └─→ EvidenceReceipt
    ↓
LedgerStorage
    ├─→ AppendOnlyLog
    └─→ MerkleAnchoring
    ↓
APILayer (출력)
```

---

## 14. MVP 범위 (Phase 1: 0-3개월) - 시뮬레이션 기반

```
NoiseChain_MVP // Phase 1 시뮬레이션 MVP 범위 (설계중)
    EdgeLayer_MVP // 가상 센서 기반 (설계중)
        VirtualSensorDriver // 가상 센서 (설계중)
            CPUTempReader // CPU 온도 (설계중)
            OSEntropyReader // OS 엔트로피 (설계중)
            SyntheticNoiseGen // 합성 노이즈 (설계중)
        RingBuffer // 버퍼 (설계중)
    TimeSyncModule_MVP // NTP만 지원 (설계중)
        NTPClient // NTP (설계중)
        TimestampGenerator // 타임스탬프 (설계중)
    AttestationPipeline_MVP // 단순 해시 (설계중)
        StatFeatures // 통계 특징만 (설계중)
        TokenHashChain // 해시 체인 (설계중)
        DeviceSignature // 서명 (설계중)
    LedgerStorage_MVP // SQLite 단일 노드 (설계중)
        SQLiteAdapter // SQLite (설계중)
        IndexStore // 인덱스 (설계중)
    VerificationLayer_MVP // 기본 검증 (설계중)
        SchemaValidator // 스키마 (설계중)
        RiskScoreCalculator // 위험점수 (설계중)
        DecisionEngine // 판정 (설계중)
    DemoScenarios // 데모 시나리오 (설계중)
        ReplayAttackDemo // 리플레이 공격 탐지 시연 (설계중)
        TamperDetectionDemo // 위조 탐지 시연 (설계중)
        TokenLifecycleDemo // 토큰 생성~검증 전체 시연 (설계중)
```

---

## 15. 로드맵

| Phase | 기간 | 목표 |
|-------|------|------|
| **P0: 시뮬레이션 MVP** | 0-3개월 | 가상 센서 기반 토큰 생성+검증+저장 |
| **P1: 하드웨어 Pilot** | 3-6개월 | 파트너사 센서 연동, 1개 산업 파일럿 |
| **P2: Scale** | 6-12개월 | 프로파일 라이브러리/앵커/운영 고도화 |
| **P3: Network** | 12-18개월 | P2P 네트워크, 마켓플레이스 |

---

*— End of Gantree Document —*
