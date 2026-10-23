"""Offline observational EEG targets. No inferred REM or causal reward."""
from collections import Counter
from statistics import mean
from pathlib import Path
from src.data.session_analysis import load_journal, finite, BANDS


def parameters(candidate):
    return {"carrier_frequency": 200., "beat_frequency": 8. if candidate == "silence" else float(candidate.split("_")[1]),
            "master_volume": 0. if candidate == "silence" else .03,
            "binaural_volume": .01, "wave_volume": 0., "rain_volume": 0., "bird_volume": 0.}


def window_mean(rows, start, end):
    selected, last = [], float("-inf")
    for row in rows:
        obs = row["observation"]
        a, b = obs.get("window_start_timestamp"), obs.get("window_end_timestamp")
        state = obs.get("state", {})
        if not (finite(a) and finite(b) and start <= a <= b <= end and a > last):
            continue
        if not (1.9 <= b-a <= 2.1 and all(finite(state.get(k)) and 0 <= state[k] <= 100 for k in BANDS)):
            continue
        if abs(sum(state[k] for k in BANDS)-100) > .1:
            continue
        selected.append(row)
        last = b
    if len(selected) < 5:
        raise ValueError("insufficient_independent_windows")
    return {k: mean(r["observation"]["state"][k] for r in selected) for k in BANDS}, [r["id"] for r in selected]


def build_trials(path):
    records, warnings, digest = load_journal(path)
    header = records[0]
    if header.get("metadata", {}).get("policy") != "AudioExperiment":
        raise ValueError("Expected AudioExperiment journal")
    if not header["audio_enabled"]:
        raise ValueError("Silent dry runs cannot provide delivered audio trials")
    rows = [r for r in records if r["type"] == "observation"]
    groups = []
    for row in rows:
        info = row["observation"].get("audio_experiment", {})
        key = info.get("block_id") if row["observation"]["ready"] else None
        if not groups or key != groups[-1][0]:
            groups.append((key, []))
        groups[-1][1].append(row)
    examples, rejected = [], Counter()
    complete = {}
    for index, (key, block) in enumerate(groups):
        if key is None:
            continue
        try:
            first = block[0]
            info = first["observation"]["audio_experiment"]
            candidate = info.get("candidate")
            if candidate not in ("silence", "beat_6", "beat_8", "beat_10"):
                raise ValueError("unknown_candidate")
            if index+1 == len(groups) or groups[index+1][0] is None:
                raise ValueError("unfinished_or_interrupted_block")
            next_row = groups[index+1][1][0]
            next_info = next_row["observation"]["audio_experiment"]
            start = first["command"].get("completed_at")
            end = next_row["command"].get("started_at")
            if not (finite(start) and finite(end) and 58 <= end-start <= 64
                    and finite(info.get("block_start")) and finite(next_info.get("block_start"))
                    and 60 <= next_info["block_start"]-info["block_start"] <= 64):
                raise ValueError("invalid_block_duration")
            if any(r.get("quality_segment") != first.get("quality_segment") for r in block+[next_row]):
                raise ValueError("quality_interruption")
            expected = parameters(candidate)
            for r in block:
                ri = r["observation"].get("audio_experiment", {})
                if ri.get("candidate") != candidate or ri.get("block_start") != info["block_start"]:
                    raise ValueError("inconsistent_block")
                if any(r.get("effective_targets", {}).get(k) != v for k, v in expected.items()):
                    raise ValueError("audio_parameters_mismatch")
            times = [r["command"].get("started_at") for r in block+[next_row]]
            if not all(finite(t) for t in times) or any(not 0 < b-a <= 4 for a,b in zip(times,times[1:])):
                raise ValueError("command_gap")
            # The transition row's EEG precedes its new audio command.
            after, after_ids = window_mean(block+[next_row], end-20, end)
            complete[index] = (after, after_ids, candidate, start, end)
            if candidate == "silence":
                before, before_ids = window_mean(block, start+10, start+30)
                kind = "silence_drift"
            else:
                prior = complete.get(index-1)
                if prior is None or prior[2] != "silence":
                    raise ValueError("missing_complete_silent_reference")
                before, before_ids = prior[:2]
                kind = "sound_after_silence"
            examples.append({"schema_version": 1, "session_id": Path(path).stem,
                "source_sha256": digest, "block_id": key, "candidate": candidate,
                "comparison": kind, "condition": header.get("metadata", {}).get("condition"),
                "before": before, "after": after,
                "delta_percentage_points": {k: after[k]-before[k] for k in BANDS},
                "before_observation_ids": before_ids, "after_observation_ids": after_ids,
                "command_start": start, "command_end": end,
                "target_kind": "observed_relative_band_change", "causal_effect": False})
        except (ValueError, KeyError, TypeError) as error:
            rejected[str(error)] += 1
    return {"source_sha256": digest, "examples": examples, "rejected_blocks": dict(rejected), "warnings": warnings}
