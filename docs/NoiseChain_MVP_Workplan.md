# NoiseChain MVP 작업 계획서

**문서 버전**: 1.0  
**작성일**: 2026-01-13  
**상태**: 진행중  
**기반 문서**: `NoiseChain_MVP_Design.md`  
**예상 기간**: 2-3일 (약 16시간)

---

## 1. 작업 개요

본 문서는 NoiseChain MVP 구현을 위한 단계별 작업 계획서입니다.  
Gantree 설계 원칙에 따라 Top-Down BFS 방식으로 작업을 분해하고,  
각 작업은 원자화 노드 수준까지 정의합니다.

---

## 2. 전체 작업 Gantree

```
NoiseChain_MVP_Implementation // MVP 구현 작업 계획 (진행중)
    Phase1_Foundation // 기반 구조 구축 (설계중)
        Task1_1_ProjectSetup // 프로젝트 초기 설정 (설계중)
        Task1_2_NoiseDataModel // 노이즈 데이터 모델 구현 (설계중)
        Task1_3_VirtualSensor // 가상 센서 드라이버 구현 (설계중)
        Task1_4_TimeSync // 시간 동기화 모듈 구현 (설계중)
    Phase2_CorePipeline // 핵심 파이프라인 구축 (설계중)
        Task2_1_FeatureExtractor // 특징 추출기 구현 (설계중)
        Task2_2_CorrelationSignature // 상관 서명 알고리즘 구현 (설계중)
        Task2_3_PoXToken // 토큰 스키마 구현 (설계중)
        Task2_4_TokenAssembler // 토큰 조립/서명 구현 (설계중)
    Phase3_Storage // 저장소 구축 (설계중)
        Task3_1_SQLiteAdapter // SQLite 어댑터 구현 (설계중)
        Task3_2_TokenStore // 토큰 저장소 구현 (설계중)
    Phase4_Verification // 검증 엔진 구축 (설계중)
        Task4_1_SchemaValidator // 스키마 검증기 구현 (설계중)
        Task4_2_SignatureVerifier // 서명 검증기 구현 (설계중)
        Task4_3_CorrelationMatcher // 상관 매처 구현 (설계중)
        Task4_4_RiskScoreCalculator // 위험점수 계산기 구현 (설계중)
        Task4_5_DecisionEngine // 판정 엔진 구현 (설계중)
    Phase5_Integration // 통합 및 데모 (설계중)
        Task5_1_EndToEndPipeline // E2E 파이프라인 연결 (설계중)
        Task5_2_DemoScenarios // 데모 시나리오 구현 (설계중)
        Task5_3_Documentation // 문서화 및 정리 (설계중)
```

---

## 3. Phase 1: 기반 구조 구축

### Task 1.1: 프로젝트 초기 설정

```
Task1_1_ProjectSetup // 프로젝트 초기 설정 (설계중)
    CreateProjectStructure // 디렉토리 구조 생성 (설계중)
        src_folder // src/ 폴더 생성
        tests_folder // tests/ 폴더 생성
        docs_folder // docs/ 이미 존재
    InitPythonProject // Python 프로젝트 초기화 (설계중)
        pyproject_toml // pyproject.toml 작성
        requirements_txt // 의존성 정의
        init_files // __init__.py 생성
    InstallDependencies // 의존성 설치 (설계중)
        numpy // 수치 계산
        scipy // 신호 처리
        psutil // 시스템 모니터링
        pynacl // Ed25519 서명
```

**산출물**:

- `src/noisechain/__init__.py`
- `pyproject.toml`
- `requirements.txt`

**예상 시간**: 30분

---

### Task 1.2: 노이즈 데이터 모델 구현

```
Task1_2_NoiseDataModel // 노이즈 데이터 모델 구현 (설계중)
    DefineSensorType // 센서 타입 열거형 (설계중)
        CPUTemp_enum // CPU 온도
        OSEntropy_enum // OS 엔트로피
        ClockJitter_enum // 클럭 지터
        SyntheticNoise_enum // 합성 노이즈
    DefineSample // 샘플 데이터 클래스 (설계중)
        timestamp_field // 타임스탬프 필드
        values_field // 센서값 딕셔너리
        validation // 유효성 검사
    DefineTimeSeries // 시계열 클래스 (설계중)
        samples_buffer // 샘플 버퍼
        add_sample // 샘플 추가 메서드
        get_window // 윈도우 추출 메서드
        to_numpy // numpy 변환 메서드
    WriteUnitTests // 단위 테스트 작성 (설계중)
        test_sensor_type // 타입 테스트
        test_sample // 샘플 테스트
        test_timeseries // 시계열 테스트
```

**산출물**:

- `src/noisechain/models/noise_data.py`
- `tests/test_noise_data.py`

**예상 시간**: 30분

---

### Task 1.3: 가상 센서 드라이버 구현

```
Task1_3_VirtualSensor // 가상 센서 드라이버 구현 (설계중)
    BaseSensorDriver // 기본 센서 드라이버 인터페이스 (설계중)
        read_method // 읽기 추상 메서드
        sensor_type_property // 센서 타입 속성
    CPUTempDriver // CPU 온도 드라이버 (설계중)
        psutil_integration // psutil 연동
        read_temperature // 온도 읽기
        fallback_random // 실패 시 랜덤
    OSEntropyDriver // OS 엔트로피 드라이버 (설계중)
        urandom_read // urandom 읽기
        byte_to_float // 바이트→실수 변환
    ClockJitterDriver // 클럭 지터 드라이버 (설계중)
        perf_counter_delta // 타이밍 차이 측정
        normalize_jitter // 정규화
    SyntheticNoiseDriver // 합성 노이즈 드라이버 (설계중)
        gaussian_noise // 가우시안 노이즈 생성
        configurable_params // 파라미터 설정
    VirtualSensorHub // 센서 허브 (설계중)
        register_sensors // 센서 등록
        collect_sample // 전체 샘플 수집
        start_collection // 연속 수집 시작
        stop_collection // 수집 중지
    WriteUnitTests // 단위 테스트 (설계중)
        test_each_driver // 드라이버별 테스트
        test_hub_collection // 허브 테스트
```

**산출물**:

- `src/noisechain/sensors/base.py`
- `src/noisechain/sensors/virtual_drivers.py`
- `src/noisechain/sensors/hub.py`
- `tests/test_sensors.py`

**예상 시간**: 1시간

---

### Task 1.4: 시간 동기화 모듈 구현

```
Task1_4_TimeSync // 시간 동기화 모듈 구현 (설계중)
    NTPClient // NTP 클라이언트 (설계중)
        ntplib_integration // ntplib 라이브러리 연동
        get_ntp_time // NTP 시간 조회
        calculate_offset // 오프셋 계산
    TimestampGenerator // 타임스탬프 생성기 (설계중)
        generate_ns // 나노초 타임스탬프 생성
        apply_offset // NTP 오프셋 적용
        monotonic_check // 단조 증가 검증
    WriteUnitTests // 단위 테스트 (설계중)
        test_ntp_sync // NTP 동기화 테스트
        test_timestamp_gen // 타임스탬프 생성 테스트
```

**산출물**:

- `src/noisechain/time/ntp_client.py`
- `src/noisechain/time/timestamp.py`
- `tests/test_time.py`

**예상 시간**: 30분

---

## 4. Phase 2: 핵심 파이프라인 구축

### Task 2.1: 특징 추출기 구현

```
Task2_1_FeatureExtractor // 특징 추출기 구현 (설계중)
    StatisticalFeatures // 통계적 특징 (설계중)
        mean_variance // 평균/분산
        skewness_kurtosis // 왜도/첨도
        entropy // 엔트로피
    FrequencyFeatures // 주파수 특징 (설계중)
        fft_compute // FFT 계산
        band_power // 대역 파워
        peak_frequency // 피크 주파수
    TemporalFeatures // 시간 특징 (설계중)
        autocorrelation // 자기상관
        zero_crossing_rate // 영교차율
    FeatureVector // 특징 벡터 조합 (설계중)
        extract_all // 전체 추출
        normalize // 정규화
        to_array // 배열 변환
    WriteUnitTests // 단위 테스트 (설계중)
        test_stat_features // 통계 테스트
        test_freq_features // 주파수 테스트
        test_combined // 통합 테스트
```

**산출물**:

- `src/noisechain/pipeline/features.py`
- `tests/test_features.py`

**예상 시간**: 1시간

---

### Task 2.2: 상관 서명 알고리즘 구현

```
Task2_2_CorrelationSignature // 상관 서명 알고리즘 구현 (설계중)
    CrossCorrelation // 교차 상관 계산 (설계중)
        scipy_correlate // scipy.signal.correlate 사용
        lag_extraction // 라그 추출
    CorrelationMatrix // 상관 행렬 구성 (설계중)
        pairwise_correlation // 센서 쌍별 계산
        matrix_build // 행렬 구성
    Quantization // 양자화 (설계중)
        min_max_normalize // MinMax 정규화
        bit_quantize // 비트 양자화
    HashGeneration // 해시 생성 (설계중)
        sha3_256_hash // SHA3-256 해시
        feature_hash_combine // 특징+해시 결합
    CorrelationSignatureClass // 메인 클래스 (설계중)
        compute_method // compute() 메서드
        verify_method // verify() 메서드
        similarity_check // 유사도 검사
    WriteUnitTests // 단위 테스트 (설계중)
        test_correlation // 상관 테스트
        test_quantization // 양자화 테스트
        test_hash // 해시 테스트
        test_verify // 검증 테스트
```

**산출물**:

- `src/noisechain/pipeline/correlation.py`
- `tests/test_correlation.py`

**예상 시간**: 2시간

---

### Task 2.3: PoXToken 스키마 구현

```
Task2_3_PoXToken // PoXToken 스키마 구현 (설계중)
    TokenHeader // 토큰 헤더 (설계중)
        header_dataclass // 데이터클래스 정의
        generate_token_id // 토큰 ID 생성
    TimeWindow // 시간창 (설계중)
        timewindow_dataclass // 데이터클래스 정의
        validate_window // 유효성 검사
    NoiseSignature // 노이즈 서명 (설계중)
        signature_dataclass // 데이터클래스 정의
    Challenge // 챌린지 (설계중)
        challenge_dataclass // 데이터클래스 정의
        generate_nonce // nonce 생성
    Integrity // 무결성 (설계중)
        integrity_dataclass // 데이터클래스 정의
    PoXTokenClass // 메인 토큰 클래스 (설계중)
        create_method // 생성 메서드
        to_bytes // 직렬화
        from_bytes // 역직렬화
        to_json // JSON 변환
        from_json // JSON 파싱
    WriteUnitTests // 단위 테스트 (설계중)
        test_serialization // 직렬화 테스트
        test_json // JSON 테스트
```

**산출물**:

- `src/noisechain/token/schema.py`
- `tests/test_token_schema.py`

**예상 시간**: 1시간

---

### Task 2.4: 토큰 조립/서명 구현

```
Task2_4_TokenAssembler // 토큰 조립/서명 구현 (설계중)
    KeyManager // 키 관리자 (설계중)
        generate_keypair // Ed25519 키 생성
        load_key // 키 로드
        save_key // 키 저장
    TokenSigner // 토큰 서명자 (설계중)
        sign_token // 토큰 서명
        get_signature_bytes // 서명 바이트 추출
    HashChain // 해시 체인 (설계중)
        compute_prev_hash // 이전 해시 계산
        link_token // 토큰 연결
    TokenAssemblerClass // 조립기 클래스 (설계중)
        assemble_method // 조립 메서드 (센서→토큰)
        mint_method // 발행 메서드
    WriteUnitTests // 단위 테스트 (설계중)
        test_signing // 서명 테스트
        test_hash_chain // 해시 체인 테스트
        test_assembly // 조립 테스트
```

**산출물**:

- `src/noisechain/token/assembler.py`
- `src/noisechain/crypto/keys.py`
- `tests/test_assembler.py`

**예상 시간**: 1시간

---

## 5. Phase 3: 저장소 구축

### Task 3.1: SQLite 어댑터 구현

```
Task3_1_SQLiteAdapter // SQLite 어댑터 구현 (설계중)
    DatabaseSchema // DB 스키마 정의 (설계중)
        tokens_table // 토큰 테이블
        receipts_table // 영수증 테이블
        create_tables_sql // 테이블 생성 SQL
    SQLiteConnection // 연결 관리 (설계중)
        connect // 연결
        close // 종료
        execute // 실행
    CRUDOperations // CRUD 연산 (설계중)
        insert_token // 토큰 삽입
        get_token_by_id // ID로 조회
        get_tokens_by_device // 장비별 조회
        get_tokens_by_time_range // 시간 범위 조회
    WriteUnitTests // 단위 테스트 (설계중)
        test_connection // 연결 테스트
        test_insert_get // 삽입/조회 테스트
```

**산출물**:

- `src/noisechain/storage/sqlite_adapter.py`
- `tests/test_sqlite.py`

**예상 시간**: 1시간

---

### Task 3.2: 토큰 저장소 구현

```
Task3_2_TokenStore // 토큰 저장소 구현 (설계중)
    TokenStoreInterface // 인터페이스 정의 (설계중)
        save_method // 저장 메서드
        load_method // 로드 메서드
        query_method // 쿼리 메서드
    TokenStoreSQLite // SQLite 구현체 (설계중)
        adapter_injection // 어댑터 주입
        implement_interface // 인터페이스 구현
    IndexManager // 인덱스 관리 (설계중)
        create_indices // 인덱스 생성
        optimize_queries // 쿼리 최적화
    WriteUnitTests // 단위 테스트 (설계중)
        test_save_load // 저장/로드 테스트
        test_query // 쿼리 테스트
```

**산출물**:

- `src/noisechain/storage/token_store.py`
- `tests/test_token_store.py`

**예상 시간**: 30분

---

## 6. Phase 4: 검증 엔진 구축

### Task 4.1: 스키마 검증기 구현

```
Task4_1_SchemaValidator // 스키마 검증기 구현 (설계중)
    VersionCheck // 버전 검사 (설계중)
        supported_versions // 지원 버전 목록
        check_version // 버전 확인
    FieldValidation // 필드 유효성 (설계중)
        required_fields // 필수 필드 검사
        type_check // 타입 검사
        range_check // 범위 검사
    SchemaValidatorClass // 검증기 클래스 (설계중)
        validate_method // validate() 메서드
        get_errors // 에러 목록 반환
    WriteUnitTests // 단위 테스트 (설계중)
        test_valid_token // 유효 토큰 테스트
        test_invalid_token // 무효 토큰 테스트
```

**산출물**:

- `src/noisechain/verification/schema_validator.py`
- `tests/test_schema_validator.py`

**예상 시간**: 30분

---

### Task 4.2: 서명 검증기 구현

```
Task4_2_SignatureVerifier // 서명 검증기 구현 (설계중)
    Ed25519Verify // Ed25519 검증 (설계중)
        load_public_key // 공개키 로드
        verify_signature // 서명 검증
    HashChainVerify // 해시 체인 검증 (설계중)
        verify_prev_hash // 이전 해시 검증
        verify_chain_link // 체인 연결 검증
    SignatureVerifierClass // 검증기 클래스 (설계중)
        verify_method // verify() 메서드
        verify_chain // 체인 검증 메서드
    WriteUnitTests // 단위 테스트 (설계중)
        test_valid_signature // 유효 서명 테스트
        test_invalid_signature // 무효 서명 테스트
        test_chain_integrity // 체인 무결성 테스트
```

**산출물**:

- `src/noisechain/verification/signature_verifier.py`
- `tests/test_signature_verifier.py`

**예상 시간**: 1시간

---

### Task 4.3: 상관 매처 구현

```
Task4_3_CorrelationMatcher // 상관 매처 구현 (설계중)
    SimilarityMetrics // 유사도 메트릭 (설계중)
        cosine_similarity // 코사인 유사도
        euclidean_distance // 유클리드 거리
    HashComparison // 해시 비교 (설계중)
        exact_match // 정확 일치
        partial_match // 부분 일치 (optional)
    CorrelationMatcherClass // 매처 클래스 (설계중)
        match_method // match() 메서드
        get_similarity_score // 유사도 점수 반환
    WriteUnitTests // 단위 테스트 (설계중)
        test_exact_match // 정확 일치 테스트
        test_similar_match // 유사 일치 테스트
        test_no_match // 불일치 테스트
```

**산출물**:

- `src/noisechain/verification/correlation_matcher.py`
- `tests/test_correlation_matcher.py`

**예상 시간**: 1.5시간

---

### Task 4.4: 위험점수 계산기 구현

```
Task4_4_RiskScoreCalculator // 위험점수 계산기 구현 (설계중)
    ScoreComponents // 점수 구성요소 (설계중)
        time_validity_score // 시간 유효성 점수
        signature_match_score // 서명 일치 점수
        correlation_score // 상관 유사도 점수
        chain_integrity_score // 체인 무결성 점수
    WeightedSum // 가중 합계 (설계중)
        configure_weights // 가중치 설정
        calculate_weighted // 가중 계산
    RiskScoreClass // 계산기 클래스 (설계중)
        calculate_method // calculate() 메서드
        get_breakdown // 상세 분석 반환
    WriteUnitTests // 단위 테스트 (설계중)
        test_all_pass // 전체 통과 (accept)
        test_partial_fail // 부분 실패 (review)
        test_all_fail // 전체 실패 (reject)
```

**산출물**:

- `src/noisechain/verification/risk_score.py`
- `tests/test_risk_score.py`

**예상 시간**: 45분

---

### Task 4.5: 판정 엔진 구현

```
Task4_5_DecisionEngine // 판정 엔진 구현 (설계중)
    ThresholdConfig // 임계값 설정 (설계중)
        accept_threshold // accept 임계값
        review_threshold // review 임계값
    DecisionLogic // 판정 로직 (설계중)
        apply_thresholds // 임계값 적용
        determine_decision // 판정 결정
    DecisionResult // 결과 구조체 (설계중)
        decision_enum // accept/review/reject
        risk_score // 위험점수
        details // 상세 정보
    DecisionEngineClass // 엔진 클래스 (설계중)
        decide_method // decide() 메서드
        create_receipt // 영수증 생성
    WriteUnitTests // 단위 테스트 (설계중)
        test_accept // accept 케이스
        test_review // review 케이스
        test_reject // reject 케이스
```

**산출물**:

- `src/noisechain/verification/decision_engine.py`
- `tests/test_decision_engine.py`

**예상 시간**: 30분

---

## 7. Phase 5: 통합 및 데모

### Task 5.1: E2E 파이프라인 연결

```
Task5_1_EndToEndPipeline // E2E 파이프라인 연결 (설계중)
    NoiseChainEngine // 메인 엔진 클래스 (설계중)
        init_components // 컴포넌트 초기화
        generate_token // 토큰 생성 (수집→발행)
        verify_token // 토큰 검증
        store_token // 토큰 저장
        query_tokens // 토큰 조회
    ConfigManager // 설정 관리 (설계중)
        load_config // 설정 로드
        validate_config // 설정 검증
    Logger // 로깅 설정 (설계중)
        configure_logging // 로깅 구성
        log_events // 이벤트 로깅
    WriteIntegrationTests // 통합 테스트 (설계중)
        test_full_pipeline // 전체 파이프라인 테스트
        test_error_handling // 에러 처리 테스트
```

**산출물**:

- `src/noisechain/engine.py`
- `src/noisechain/config.py`
- `tests/test_integration.py`

**예상 시간**: 1시간

---

### Task 5.2: 데모 시나리오 구현

```
Task5_2_DemoScenarios // 데모 시나리오 구현 (설계중)
    ReplayAttackDemo // 리플레이 공격 데모 (설계중)
        scenario_script // 시나리오 스크립트
        expected_result_reject // 예상 결과: reject
    TamperDetectionDemo // 위조 탐지 데모 (설계중)
        tamper_feature // 특징 변조 로직
        expected_result_reject // 예상 결과: reject
    TokenLifecycleDemo // 생명주기 데모 (설계중)
        full_lifecycle // 전체 생명주기
        expected_result_accept // 예상 결과: accept
    DemoRunner // 데모 실행기 (설계중)
        run_all_demos // 전체 데모 실행
        print_results // 결과 출력
        generate_report // 보고서 생성
```

**산출물**:

- `demos/replay_attack_demo.py`
- `demos/tamper_detection_demo.py`
- `demos/token_lifecycle_demo.py`
- `demos/run_all.py`

**예상 시간**: 1시간

---

### Task 5.3: 문서화 및 정리

```
Task5_3_Documentation // 문서화 및 정리 (설계중)
    UpdateREADME // README 업데이트 (설계중)
        installation_guide // 설치 가이드
        usage_examples // 사용 예시
        api_overview // API 개요
    CodeComments // 코드 주석 정리 (설계중)
        docstrings // 독스트링 작성
        inline_comments // 인라인 주석
    TestCoverage // 테스트 커버리지 (설계중)
        run_coverage // 커버리지 실행
        generate_report // 리포트 생성
    FinalReview // 최종 검토 (설계중)
        code_review // 코드 리뷰
        update_design_docs // 설계 문서 업데이트
```

**산출물**:

- `README.md`
- 코드 주석 완성
- 테스트 커버리지 리포트

**예상 시간**: 1시간

---

## 8. 작업 진행 체크리스트

### Phase 1: 기반 구조 (예상 2시간)

- [ ] Task 1.1: 프로젝트 초기 설정
- [ ] Task 1.2: 노이즈 데이터 모델 구현
- [ ] Task 1.3: 가상 센서 드라이버 구현

### Phase 2: 핵심 파이프라인 (예상 5시간)

- [ ] Task 2.1: 특징 추출기 구현
- [ ] Task 2.2: 상관 서명 알고리즘 구현
- [ ] Task 2.3: PoXToken 스키마 구현
- [ ] Task 2.4: 토큰 조립/서명 구현

### Phase 3: 저장소 (예상 1.5시간)

- [ ] Task 3.1: SQLite 어댑터 구현
- [ ] Task 3.2: 토큰 저장소 구현

### Phase 4: 검증 엔진 (예상 4.5시간)

- [ ] Task 4.1: 스키마 검증기 구현
- [ ] Task 4.2: 서명 검증기 구현
- [ ] Task 4.3: 상관 매처 구현
- [ ] Task 4.4: 위험점수 계산기 구현
- [ ] Task 4.5: 판정 엔진 구현

### Phase 5: 통합 및 데모 (예상 3시간)

- [ ] Task 5.1: E2E 파이프라인 연결
- [ ] Task 5.2: 데모 시나리오 구현
- [ ] Task 5.3: 문서화 및 정리

---

## 9. 예상 디렉토리 구조

```
NoiseChain/
├── docs/
│   ├── NoiseChain_Specification.md
│   ├── NoiseChain_Gantree.md
│   ├── NoiseChain_MVP_Design.md
│   └── NoiseChain_MVP_Workplan.md  # 본 문서
├── src/
│   └── noisechain/
│       ├── __init__.py
│       ├── engine.py              # 메인 엔진
│       ├── config.py              # 설정 관리
│       ├── models/
│       │   └── noise_data.py      # 데이터 모델
│       ├── sensors/
│       │   ├── base.py            # 센서 베이스
│       │   ├── virtual_drivers.py # 가상 드라이버
│       │   └── hub.py             # 센서 허브
│       ├── pipeline/
│       │   ├── features.py        # 특징 추출
│       │   └── correlation.py     # 상관 서명
│       ├── token/
│       │   ├── schema.py          # 토큰 스키마
│       │   └── assembler.py       # 토큰 조립
│       ├── crypto/
│       │   └── keys.py            # 키 관리
│       ├── storage/
│       │   ├── sqlite_adapter.py  # SQLite
│       │   └── token_store.py     # 토큰 저장소
│       └── verification/
│           ├── schema_validator.py
│           ├── signature_verifier.py
│           ├── correlation_matcher.py
│           ├── risk_score.py
│           └── decision_engine.py
├── tests/
│   ├── test_noise_data.py
│   ├── test_sensors.py
│   ├── test_features.py
│   ├── test_correlation.py
│   ├── test_token_schema.py
│   ├── test_assembler.py
│   ├── test_sqlite.py
│   ├── test_token_store.py
│   ├── test_schema_validator.py
│   ├── test_signature_verifier.py
│   ├── test_correlation_matcher.py
│   ├── test_risk_score.py
│   ├── test_decision_engine.py
│   └── test_integration.py
├── demos/
│   ├── replay_attack_demo.py
│   ├── tamper_detection_demo.py
│   ├── token_lifecycle_demo.py
│   └── run_all.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 10. 작업 시작 가이드

**첫 번째 세션** (Phase 1 완료 목표):

1. Task 1.1: 프로젝트 구조 생성
2. Task 1.2: 데이터 모델 구현
3. Task 1.3: 가상 센서 구현

**다음 세션**: Phase 2 진행

---

*— End of Work Plan —*
