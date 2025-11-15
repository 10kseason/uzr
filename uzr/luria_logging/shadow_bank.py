# luria_logging/shadow_bank.py
from datetime import datetime
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger_luria import append_jsonl


def log_sb_event(step, action, count, stats):
    """Log shadow bank event to JSONL file.

    Args:
        step: Current training step
        action: Action type (promote, decay, skip, commit_shadow, merge, near_merge)
        count: Number of items affected by this event
        stats: Statistics dictionary with thresholds, sizes, distributions, etc.
    """
    append_jsonl("logus-luria/shadow_bank_events.jsonl", {
        "ts": datetime.utcnow().isoformat(),
        "step": int(step),
        "action": action,
        "count": int(count),
        "stats": stats
    })
