# luria_logging/metrics.py
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger_luria import CsvWriter


def safe_div(a, b):
    """Safe division that returns 0.0 when denominator is 0."""
    return float(a) / float(b) if b != 0 else 0.0


# Initialize CSV writer for shadow bank statistics
_sb_snap = None


def get_sb_snapshot_writer():
    """Get or create shadow bank snapshot CSV writer."""
    global _sb_snap
    if _sb_snap is None:
        _sb_snap = CsvWriter("logus-luria/shadow_bank_stats.csv", fieldnames=[
            "step", "stage", "shadow_size", "shadow_mean_age", "shadow_age_p95",
            "shadow_surprise_mean", "shadow_surprise_p75",
            "promote_ratio", "decay_freq", "skip_count",
            "commit_count", "merge_count", "near_merge_count",
            "mem_size", "wm_curr", "recall_topk"
        ])
    return _sb_snap


def log_sb_snapshot(step, stage, bank, counters, wm_curr, recall_topk):
    """Log shadow bank snapshot statistics to CSV.

    Args:
        step: Current training step
        stage: Current stage name
        bank: Shadow bank list/queue object
        counters: Dictionary with tick and total counters
        wm_curr: Current working memory size
        recall_topk: Current recall top-k value
    """
    writer = get_sb_snapshot_writer()

    # Calculate promote ratio
    promote_ratio = safe_div(
        counters.get("promote_total", 0),
        counters.get("promote_total", 0) + counters.get("skip_total", 0)
    )

    # Calculate shadow bank statistics
    shadow_size = len(bank) if hasattr(bank, '__len__') else 0
    shadow_mean_age = 0.0
    shadow_age_p95 = 0.0
    shadow_surprise_mean = 0.0
    shadow_surprise_p75 = 0.0

    if hasattr(bank, 'mean_age'):
        shadow_mean_age = bank.mean_age()
    elif isinstance(bank, list) and shadow_size > 0:
        # Calculate manually if bank is a list
        ages = [step - getattr(item, 'step', step) for item in bank]
        shadow_mean_age = sum(ages) / len(ages) if ages else 0.0
        if ages:
            ages_sorted = sorted(ages)
            p95_idx = min(len(ages_sorted) - 1, int(0.95 * len(ages_sorted)))
            shadow_age_p95 = ages_sorted[p95_idx]

    if hasattr(bank, 'age_percentile'):
        shadow_age_p95 = bank.age_percentile(95)

    if hasattr(bank, 'surprise_mean'):
        shadow_surprise_mean = bank.surprise_mean()
    elif isinstance(bank, list) and shadow_size > 0:
        # Calculate manually if bank is a list
        surprises = [getattr(item, '_surprise', 0.0) for item in bank]
        surprises = [s for s in surprises if s is not None]
        if surprises:
            shadow_surprise_mean = sum(surprises) / len(surprises)
            surprises_sorted = sorted(surprises)
            p75_idx = min(len(surprises_sorted) - 1, int(0.75 * len(surprises_sorted)))
            shadow_surprise_p75 = surprises_sorted[p75_idx]

    if hasattr(bank, 'surprise_percentile'):
        shadow_surprise_p75 = bank.surprise_percentile(75)

    writer.write_row({
        "step": step,
        "stage": stage,
        "shadow_size": shadow_size,
        "shadow_mean_age": round(shadow_mean_age, 4),
        "shadow_age_p95": round(shadow_age_p95, 4),
        "shadow_surprise_mean": round(shadow_surprise_mean, 4),
        "shadow_surprise_p75": round(shadow_surprise_p75, 4),
        "promote_ratio": round(promote_ratio, 4),
        "decay_freq": counters.get("decay_tick", 0),
        "skip_count": counters.get("skip_tick", 0),
        "commit_count": counters.get("commit_tick", 0),
        "merge_count": counters.get("merge_tick", 0),
        "near_merge_count": counters.get("near_merge_tick", 0),
        "mem_size": counters.get("mem_size_curr", 0),
        "wm_curr": wm_curr,
        "recall_topk": recall_topk
    })
