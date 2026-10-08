import argparse
import json

from . import __version__


def main() -> None:
    ap = argparse.ArgumentParser(prog="dq-agent")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="investigate a CSV/Parquet file")
    r.add_argument("path")
    sc = sub.add_parser("score", help="score a run against the injected ground truth")
    sc.add_argument("report")
    sc.add_argument("csv")
    g = sub.add_parser("gen-data", help="write synthetic orders data with injected anomalies")
    g.add_argument("out_dir", nargs="?", default="data/raw")
    args = ap.parse_args()

    if args.cmd == "gen-data":
        from .synth.orders import generate

        print(generate(args.out_dir))
    elif args.cmd == "run":
        from .agent import run
        from .llm import get_llm

        rep = run(args.path, get_llm(), verbose=True)
        for f in rep["findings"]:
            print(
                f"[{f.get('verdict')}] {f.get('hypothesis')} (verified={f.get('verified_count')})"
            )
        print(json.dumps({k: rep[k] for k in ("run_id", "tool_calls", "tokens")}))
    elif args.cmd == "score":
        from .eval.score import score_report

        print(json.dumps(score_report(args.report, args.csv), indent=2))
    else:
        ap.print_help()
