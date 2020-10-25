from typing import Mapping

TRAINING_PARAMS: Mapping[str, float] = {
    "lr": 1.0,
    "dim": 100,
    "minCount": 1,
    "wordNGrams": 2,
    "minn": 2,
    "maxn": 5,
}
QUANTIZE: bool = True
