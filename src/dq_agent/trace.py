import json
import sys
import time
from pathlib import Path


class Trace:
    """Append-only JSONL audit log for one run."""

    def __init__(self, runs_dir: str | Path, run_id: str, verbose: bool = False):
        self.verbose, self.t0 = verbose, time.time()
        self.dir = Path(runs_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"

    def log(self, event: str, **data) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "event": event, **data}, default=str))
            f.write("\n")
        if self.verbose:
            detail = data.get("sql") or data.get("reason") or data.get("error") or ""
            tok = f" tokens={data['tokens']}" if "tokens" in data else ""
            print(f"[{time.time() - self.t0:6.1f}s] {event}{tok} {str(detail)[:100]}",
                  file=sys.stderr, flush=True)  # fmt: skip
