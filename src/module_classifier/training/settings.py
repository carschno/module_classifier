from typing import Dict

TRAINING_PARAMS: Dict[str, float] = {
    "lr": 1.0,
    "dim": 100,
    "minCount": 1,
    "wordNgrams": 2,
    "minn": 2,
    "maxn": 5,
    "epoch": 20,
}
QUANTIZE: bool = True
