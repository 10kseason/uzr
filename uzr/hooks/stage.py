# hooks/stage.py
from datetime import datetime
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger_luria import append_jsonl


def log_stage_event(step, stage_name, reason, meta=None):
    """Log stage transition event to JSONL file.

    Args:
        step: Current training step
        stage_name: Name of the stage (e.g., 'warmup', 'rules_simple', 'eval', etc.)
        reason: Reason for stage transition
        meta: Optional metadata dictionary
    """
    append_jsonl("logus-luria/stage_events.jsonl", {
        "ts": datetime.utcnow().isoformat(),
        "step": int(step),
        "event": "stage_change",
        "stage": stage_name,
        "reason": reason,
        "meta": meta or {}
    })
