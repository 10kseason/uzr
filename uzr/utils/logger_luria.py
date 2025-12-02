# utils/logger_luria.py
import json
import os
import csv
from datetime import datetime


def _ensure_dir(path):
    """Ensure parent directory exists for given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_jsonl(path, obj):
    """Append JSON object as line to JSONL file."""
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class CsvWriter:
    """CSV writer that automatically creates file with header if needed."""

    def __init__(self, path, fieldnames):
        _ensure_dir(path)
        self.path = path
        self.fieldnames = fieldnames
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write_row(self, row):
        """Write a single row to CSV file."""
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
