"""Create a local JSON + Markdown report without running EEG or audio."""
import argparse
import json
from pathlib import Path
from uuid import uuid4
from src.data.session_analysis import analyze_session, markdown_report

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', nargs='?', type=Path, help='Journal JSONL file')
    parser.add_argument('--latest', action='store_true', help='Latest local journal in data/sessions')
    args = parser.parse_args()
    if bool(args.journal) == bool(args.latest):
        parser.error('Specify a journal path OR --latest')
    source = args.journal
    if args.latest:
        files = list((ROOT/'data/sessions').glob('*.jsonl'))
        if not files:
            parser.error('No session journals yet; run run_sleep_observer.py first')
        source = max(files, key=lambda p: p.stat().st_mtime_ns)
    try:
        report = analyze_session(source)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot analyze session: {error}\n')
    directory = ROOT/'data/sessions/reports'/f'{source.stem}_{uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (directory/'report.md').write_text(markdown_report(report), encoding='utf-8')
    print(f"Observations: {report['observations']} | usable: {report['valid_observations']} | rejected: {report['rejected_observations']}")
    print(f"Audio enabled: {report['audio_enabled']} | status: {report['status']}")
    for warning in report['warnings']:
        print('NOTE:', warning)
    print('Report:', directory/'report.md')


if __name__ == '__main__':
    main()
