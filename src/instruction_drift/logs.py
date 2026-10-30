"""Structured JSON-lines logging, one log file per run."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from instruction_drift.schemas import utc_now

_STD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("instruction_drift")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    logger.propagate = False
    return logger
