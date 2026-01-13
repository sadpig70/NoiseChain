# NoiseChain MVP Design Document

**Version**: 1.0  
**Date**: 2026-01-13  
**Status**: Implemented  
**Scope**: Phase 0 Simulation MVP (0-3 months)  
**Environment**: Simulation-based (Virtual noise using CPU/OS entropy)

[🇰🇷 Korean Version (한국어)](NoiseChain_MVP_Design_ko.md)

---

## 1. MVP Goals

- Complete the PoXToken generation & verification pipeline based on virtual sensors.
- Validate Replay Attack Detection and Anti-Spoofing algorithms.
- Demonstrate core value through end-to-end scenarios.

---

## 2. MVP Full Gantree

```
NoiseChain_MVP // Phase 0 Simulation MVP (Implemented)
    EdgeLayer_MVP // Noise collection based on virtual sensors (Implemented)
        VirtualSensorDriver // Virtual Sensor Driver (Implemented)
        NoiseDataModel // Noise Data Model (Implemented)
        RingBuffer // Circular Buffer (Implemented)
    TimeSyncModule_MVP // NTP Time Synchronization (Implemented)
        NTPClient // NTP Client (Implemented)
        TimestampGenerator // Timestamp Generation (Implemented)
    AttestationPipeline_MVP // Core PoX Generation (Implemented)
        FeatureExtractor_MVP // Feature Extraction (Implemented)
        CorrelationSignature // Correlation Signature Algorithm (Implemented)
        PoXToken_v1 // Token Schema (Implemented)
        TokenAssembler // Token Assembly/Signing (Implemented)
    LedgerStorage_MVP // SQLite Storage (Implemented)
        SQLiteAdapter // SQLite Interface (Implemented)
        TokenStore // Token Storage (Implemented)
        IndexStore // Index Management (Implemented)
    VerificationLayer_MVP // Verification Engine (Implemented)
        SchemaValidator // Schema Validation (Implemented)
        SignatureVerifier // Signature Verification (Implemented)
        CorrelationMatcher // Correlation Pattern Matching (Implemented)
        RiskScoreCalculator // Risk Score Calculation (Implemented)
        DecisionEngine // Decision Logic (Implemented)
    DemoScenarios // Demo Scenarios (Implemented)
        ReplayAttackDemo // Replay Attack Demonstration (Implemented)
        TamperDetectionDemo // Tamper Detection Demonstration (Implemented)
        TokenLifecycleDemo // Full Lifecycle Demonstration (Implemented)
```

---

## 3. Virtual Sensor Data Model (NoiseDataModel)

### 3.1 Gantree Details

```
NoiseDataModel // Virtual Sensor Data Model (Implemented)
    SensorType // Sensor Type Enumeration (Implemented)
        CPUTemp // CPU Temperature (Implemented)
        OSEntropy // OS Entropy (Implemented)
        ClockJitter // Clock Jitter (Implemented)
        SyntheticNoise // Synthetic Noise (Implemented)
    Sample // Single Sample (Implemented)
        timestamp_ns // Nanosecond Timestamp (Implemented)
        sensor_values // Sensor Values Map (Implemented)
    TimeSeries // Time Series Buffer (Implemented)
        samples // Sample Vector (Implemented)
        sampling_rate_hz // Sampling Rate (Implemented)
        window_size // Window Size (Implemented)
```

### 3.2 Implemented Sensors

| Sensor Type | Data Source | Implementation |
|-------------|-------------|----------------|
| CPUTemp | `psutil.sensors_temperatures()` | System API call |
| OSEntropy | `/dev/urandom` or `os.urandom()` | Byte reading |
| ClockJitter | `time.perf_counter_ns()` delta | Timing measurement |
| SyntheticNoise | `numpy.random.normal()` | Gaussian noise |

---

## 4. PoXToken v1 Schema

### 4.1 Schema Definition

```rust
// PoXToken v1 Structure (Total ~199 bytes)
struct PoXToken_v1 {
    // Header
    token_id: [u8; 32],       // 32 bytes - SHA3-256(content)
    version: u8,              // 1 byte - 0x01
    device_id: String,        // 16 bytes (UUID bytes)
    profile_id: String,       // "default"
    timestamp_ns: i64,        // 8 bytes - nanoseconds since epoch
    
    // NoiseSignature
    fingerprint: {
        feature_vector: bytes,    // 64 bytes
        correlation_hash: bytes,  // 32 bytes
        sensor_count: u8,         // 1 byte
        sample_count: u16         // 2 bytes
    },
    
    // Risk Score
    risk_score: f64,          // 8 bytes
    
    // Integrity
    signature: [u8; 64]       // 64 bytes - Ed25519 signature
}
```

### 4.2 JSON Representation (Example)

```json
{
  "version": 1,
  "node_id": "04a1... (16 bytes)",
  "timestamp_ns": 1705132991123456789,
  "fingerprint": {
    "feature_vector": "base64...",
    "correlation_hash": "sha3-256...",
    "sensor_count": 4,
    "sample_count": 256
  },
  "risk_score": 15.5,
  "signature": "ed25519_sig..."
}
```

---

## 5. Correlation Signature Algorithm

### 5.1 Algorithm Definition

```python
class CorrelationSignature:
    # Parameters
    MAX_LAG = 10          # Max lag (samples)
    LAG_STEP = 1          # Lag step
    QUANT_BITS = 8        # Quantization bits
    SIMILARITY_THRESHOLD = 0.85  # Verification threshold
    
    def compute(self, sensor_data):
        # 1. Compute Cross-Correlation Matrix (N x N x L)
        # 2. Normalize (MinMax)
        # 3. Extract Feature Vector (Upper Triangle + Diagonal)
        # 4. Quantize (8-bit)
        # 5. Generate SHA3-256 Hash
        return feature_vector, correlation_hash
```

---

## 6. RiskScore Calculation

### 6.1 Logic

```python
class RiskScoreCalculator:
    # Weights (Total = 1.0)
    WEIGHTS = {
        'time_validity': 0.20,
        'signature_match': 0.25,
        'correlation_sim': 0.40,  # Most important
        'chain_integrity': 0.15
    }
    
    # Thresholds
    THRESHOLDS = {
        'accept_max': 30.0,   # risk < 30 -> accept
        'review_max': 70.0,   # 30 <= risk < 70 -> review
        # risk >= 70 -> reject
    }
```

---

## 7. Demo Scenarios

### 7.1 Implemented Scenarios

#### Demo 1: Token Generation & Verification

1. Collect noise from virtual sensors.
2. Mint PoXToken.
3. Verify signature, schema, and risk score.
4. **Result**: Valid token (Risk Score < 30).

#### Demo 2: Tamper Detection (Simulated)

1. Generate valid token.
2. Tamper with `feature_vector` or `timestamp`.
3. Submit for verification.
4. **Result**: Invalid signature or High Risk Score.

---

## 8. Implementation Order (Completed)

| Order | Node | Complexity | Status |
|-------|------|------------|--------|
| 1 | NoiseDataModel | Low | ✅ Done |
| 2 | VirtualSensorDriver | Mid | ✅ Done |
| 3 | TimestampGenerator | Low | ✅ Done |
| 4 | FeatureExtractor | Mid | ✅ Done |
| 5 | CorrelationSignature | High | ✅ Done |
| 6 | PoXToken_v1 | Mid | ✅ Done |
| 7 | TokenAssembler | Mid | ✅ Done |
| 8 | SQLiteAdapter | Mid | ✅ Done |
| 9 | SignatureVerifier | Mid | ✅ Done |
| 10 | CorrelationMatcher | High | ✅ Done |
| 11 | RiskScoreCalculator | Mid | ✅ Done |
| 12 | DecisionEngine | Low | ✅ Done |
| 13 | DemoScenarios | Low | ✅ Done |

**Total Time**: ~12 hours (Actual)

---

## 9. Tech Stack (MVP)

| Area | Technology | Reason |
|------|------------|--------|
| Language | Python 3.11+ | Rapid prototyping |
| Crypto | `PyNaCl` (Ed25519) | High performance signing |
| Signal | `numpy`, `scipy` | Correlation computation |
| Storage | `sqlite3` | Zero-configuration DB |
| Time | `ntplib` | Network time sync |
| Sensor | `psutil` | System info access |

---

## 10. Verification Checklist

### Functional Verification

- [x] Noise collection from virtual sensors
- [x] Feature extraction algorithm
- [x] Correlation signature generation/verification
- [x] PoXToken serialization/deserialization
- [x] SQLite save/query
- [x] RiskScore calculation
- [x] Replay attack detection (via Timestamp/Nonce)
- [x] Tamper detection (via Signature)

### Performance Goals

- [x] Token Generation < 100ms (Actual: ~10ms)
- [x] Token Verification < 50ms (Actual: ~2ms)
- [x] Storage Query < 10ms (Actual: ~1ms)

---

*— End of MVP Design Document —*
