"""NoiseChain 파이프라인 패키지 - 특징 추출 및 상관 서명"""

from noisechain.pipeline.correlation import (
    CorrelationConfig,
    CorrelationResult,
    CorrelationSignature,
    compute_correlation_signature,
)
from noisechain.pipeline.features import (
    FeatureConfig,
    FeatureExtractor,
    FeatureType,
    FeatureVector,
    extract_features,
)

__all__ = [
    # Features
    "FeatureType",
    "FeatureConfig",
    "FeatureVector",
    "FeatureExtractor",
    "extract_features",
    # Correlation
    "CorrelationConfig",
    "CorrelationResult",
    "CorrelationSignature",
    "compute_correlation_signature",
]
