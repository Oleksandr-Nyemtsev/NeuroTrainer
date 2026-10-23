"""Export complete audio blocks locally without devices or network."""
import argparse
import json
from pathlib import Path
from src.data.audio_trials import build_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "data/sessions"
    if bool(args.journal) == args.latest:
        parser.error("Choose a journal OR --latest")
    try:
        if args.latest:
            paths = list(root.glob("*.jsonl"))
            if not paths:
                raise ValueError("No journals")
            source = max(paths, key=lambda p: p.stat().st_mtime_ns)
        else:
            source = args.journal
        report = build_trials(source)
        output = args.output or root / "trials" / (source.stem + "_" + report["source_sha256"][:8] + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print("Examples:", len(report["examples"]), "| Rejected blocks:", report["rejected_blocks"])
        print("Output:", output)
        print("Observed changes only; not REM labels or proven sound effects.")
    except (OSError, ValueError) as error:
        parser.exit(1, str(error)+"\n")


if __name__ == "__main__":
    main()
