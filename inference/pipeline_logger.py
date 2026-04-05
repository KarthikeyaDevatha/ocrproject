"""
Pipeline Logger for Adaptive OCR Agent.

Structured logging for every decision, retry, and result in the pipeline.
Outputs to both console and a list of log entries for UI display.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class LogLevel(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


@dataclass
class LogEntry:
    """Single log entry."""
    timestamp: float
    level: str
    message: str
    stage: str = ""  # e.g., "PREPROCESS", "DECISION", "OCR", "RETRY"


class PipelineLogger:
    """
    Structured pipeline logger.
    
    Captures every decision point for:
    1. Console output during development
    2. Streamlit log viewer in the UI
    3. Debug trace for evaluation
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.entries: List[LogEntry] = []
        self._start_time = time.time()

    def _elapsed(self) -> str:
        return f"{time.time() - self._start_time:.3f}s"

    def log(self, message: str, level: str = "INFO", stage: str = ""):
        """Log a message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            stage=stage
        )
        self.entries.append(entry)
        
        if self.verbose:
            prefix = f"[{level}]"
            if stage:
                prefix += f" [{stage}]"
            print(f"{prefix} {message}")

    def info(self, message: str, stage: str = ""):
        self.log(message, "INFO", stage)

    def warn(self, message: str, stage: str = ""):
        self.log(message, "WARN", stage)

    def error(self, message: str, stage: str = ""):
        self.log(message, "ERROR", stage)

    def debug(self, message: str, stage: str = ""):
        self.log(message, "DEBUG", stage)

    def log_decision(self, engine: str, profile: str, reason: str, features: dict):
        """Log an engine decision."""
        feat_str = ", ".join(f"{k}={v}" for k, v in features.items() 
                            if k in ["blur_score", "math_density", "std_intensity", "line_count"])
        self.info(f"Features: {feat_str}", "DECISION")
        self.info(f"Profile: {profile.upper()}", "DECISION")
        self.info(f"Engine: {engine} ({reason})", "DECISION")

    def log_confidence(self, line_idx: int, composite: float, tag: str,
                       engine: str, retried: bool = False):
        """Log confidence result for a line."""
        retry_note = " (after retry)" if retried else ""
        self.info(
            f"Line {line_idx} → composite={composite:.4f} → {tag} "
            f"[{engine}]{retry_note}",
            "CONFIDENCE"
        )

    def log_retry(self, line_idx: int, from_engine: str, to_engine: str,
                  original_score: float):
        """Log a retry decision."""
        self.warn(
            f"Line {line_idx}: {from_engine} confidence={original_score:.4f} → "
            f"retrying with {to_engine}",
            "RETRY"
        )

    def log_postprocess(self, corrections: int, merges: int):
        """Log post-processing stats."""
        self.info(
            f"Post-processing: {corrections} spell corrections, {merges} line merges",
            "POSTPROCESS"
        )

    def log_final(self, tag: str, total_time: float, line_count: int):
        """Log final pipeline result."""
        self.info(f"Final tag: {tag}", "OUTPUT")
        self.info(f"Lines processed: {line_count}", "OUTPUT")
        self.info(f"Total time: {total_time:.3f}s", "OUTPUT")

    def reset(self):
        """Reset logger for new pipeline run."""
        self.entries.clear()
        self._start_time = time.time()

    def get_entries(self) -> List[dict]:
        """Get all log entries as dicts (for UI display)."""
        return [
            {
                "level": e.level,
                "stage": e.stage,
                "message": e.message,
                "elapsed": f"{e.timestamp - self._start_time:.3f}s"
            }
            for e in self.entries
        ]

    def format_for_display(self) -> str:
        """Format all entries as a readable string for Streamlit."""
        lines = []
        for e in self.entries:
            elapsed = f"{e.timestamp - self._start_time:.3f}s"
            prefix = f"[{e.level}]"
            if e.stage:
                prefix += f" [{e.stage}]"
            lines.append(f"{elapsed} {prefix} {e.message}")
        return "\n".join(lines)
