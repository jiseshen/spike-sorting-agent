"""Stage 1: Simulate extracellular recordings with MEArec."""
from .setting import SettingConfig
from .generator import generate_recording
from .overcluster import overcluster_recording

__all__ = ["SettingConfig", "generate_recording", "overcluster_recording"]
