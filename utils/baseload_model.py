"""Fail closed on incompatible/mixed baseload model artifacts in archive mode."""
from __future__ import annotations

import json

from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract
from utils.archive_quality import LEAF_ENERGY_ESTIMATE_POLICY


def validate_baseload_model(features: object, metadata: object, embedded_contract: object,
                            model_feature_names: object) -> None:
    message = 'Incompatible baseload model feature contract; retrain with the current feature set before archive use'
    expected = baseload_feature_contract()
    if features != list(BASELOAD_FEATURES) or model_feature_names != list(BASELOAD_FEATURES):
        raise ValueError(message)
    if not isinstance(metadata, dict) or metadata.get('feature_contract') != expected:
        raise ValueError(message)
    if not isinstance(embedded_contract, str):
        raise ValueError(message)
    try:
        embedded = json.loads(embedded_contract)
    except (ValueError, TypeError) as error:
        raise ValueError(message) from error
    if embedded != expected:
        raise ValueError(message)
    training = metadata.get('archive_training')
    if not isinstance(training, dict) or training.get('leaf_energy_estimate_policy') != LEAF_ENERGY_ESTIMATE_POLICY:
        raise ValueError('Incompatible Leaf-energy estimate policy; retrain on archive data before archive use')
