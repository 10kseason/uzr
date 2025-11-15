# luria_logging/dedup.py
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger_luria import CsvWriter


# Initialize CSV writer for deduplication statistics
_dedup = None


def get_dedup_writer():
    """Get or create deduplication CSV writer."""
    global _dedup
    if _dedup is None:
        _dedup = CsvWriter("logus-luria/dedup_stats.csv", fieldnames=[
            "step", "stage", "dup_rate", "dup_count", "checked", "dup_sim_thr",
            "avg_sim_candidates", "p95_sim_candidates"
        ])
    return _dedup


def log_dedup(step, stage, checked, dup_count, sim_thr, sim_stats):
    """Log deduplication statistics to CSV.

    Args:
        step: Current training step
        stage: Current stage name
        checked: Number of items checked for duplicates
        dup_count: Number of duplicates found
        sim_thr: Similarity threshold used for duplicate detection
        sim_stats: Dictionary with mean and p95 similarity statistics
    """
    writer = get_dedup_writer()

    dup_rate = round(dup_count / max(1, checked), 4)

    writer.write_row({
        "step": step,
        "stage": stage,
        "dup_rate": dup_rate,
        "dup_count": dup_count,
        "checked": checked,
        "dup_sim_thr": sim_thr,
        "avg_sim_candidates": round(sim_stats.get("mean", 0.0), 4),
        "p95_sim_candidates": round(sim_stats.get("p95", 0.0), 4)
    })
