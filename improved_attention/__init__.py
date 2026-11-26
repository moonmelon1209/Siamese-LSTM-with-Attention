"""Masked attention pooling for the multilingual MaLSTM experiment."""
from pathlib import Path


DEFAULT_RUN = Path(__file__).resolve().parent / "artifacts" / "multilingual_attention"
MODEL_TYPE = "multilingual_attention_pooling"
