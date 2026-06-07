"""Shared setup output helpers."""


def empty_setup(setup_type: str, setup_name: str) -> dict:
    return {
        "setup_type": setup_type,
        "setup_name": setup_name,
        "bias": "neutral",
        "is_valid_setup": False,
        "confidence_score": 0,
        "current_price": 0,
        "resistance_level": 0,
        "stop_loss": 0,
        "atr_14": 0,
        "volume_ratio": 0,
        "details": "",
    }