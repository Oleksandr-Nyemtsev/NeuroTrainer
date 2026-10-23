"""Create local personal references from two completed raw awake-control sessions."""
import argparse
import json
from pathlib import Path

from src.signal.personal_baseline import build_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eyes-open',type=Path,required=True,help='Completed awake-eyes-open journal JSONL')
    parser.add_argument('--eyes-closed',type=Path,required=True,help='Completed awake-eyes-closed journal JSONL')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing personal baseline')
    profile = build_profile(args.eyes_open,args.eyes_closed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(profile,handle,indent=2,allow_nan=False)
    print('Personal baseline:',args.output)
    for condition,reference in profile['conditions'].items():
        print(f"{condition}: {reference['accepted_windows']} non-overlapping windows, {reference['accepted_seconds_nominal']} nominal seconds")
    print('Diagnostic only. No model fitting, stage labels or audio-policy changes.')


if __name__ == '__main__':
    main()
