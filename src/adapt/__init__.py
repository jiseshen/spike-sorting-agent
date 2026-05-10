"""Stage 4: Few-shot adaptation from trajectory demonstrations."""
from .sampler import sample_channels, TrainEvalSplit
from .evaluator import evaluate_on_channels

__all__ = ["sample_channels", "TrainEvalSplit", "evaluate_on_channels"]
