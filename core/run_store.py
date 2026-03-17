import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from providers.base import AnalysisResult

DEFAULT_RUNS_DIR = Path("data/runs")


@dataclass
class RunRecord:
    run_id: str
    name: str
    timestamp: str
    element: str
    mllms: list
    pair_count: int
    total_calls: int
    summary: dict
    results: list


def _result_to_dict(result: AnalysisResult) -> dict:
    return {
        "pair_id": result.pair_id,
        "mllm": result.mllm,
        "element": result.element,
        "description_a": result.description_a,
        "description_b": result.description_b,
        "presence_a": result.presence_a,
        "presence_b": result.presence_b,
        "match": result.match,
        "reasoning": result.reasoning,
        "tokens_used": result.tokens_used,
        "latency_ms": result.latency_ms,
        "raw_response": result.raw_response,
        "timestamp": result.timestamp.isoformat() if result.timestamp else None,
        "error": result.error,
        "ground_truth": result.ground_truth,
    }


def _dict_to_result(d: dict) -> AnalysisResult:
    ts = d.get("timestamp")
    result = AnalysisResult(
        pair_id=d.get("pair_id", ""),
        mllm=d.get("mllm", ""),
        element=d.get("element", ""),
        description_a=d.get("description_a"),
        description_b=d.get("description_b"),
        presence_a=d.get("presence_a"),
        presence_b=d.get("presence_b"),
        reasoning=d.get("reasoning"),
        tokens_used=d.get("tokens_used"),
        latency_ms=d.get("latency_ms"),
        raw_response=d.get("raw_response"),
        error=d.get("error"),
        ground_truth=d.get("ground_truth"),
    )
    if ts:
        try:
            result.timestamp = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass
    return result


def _compute_summary(results: list[AnalysisResult]) -> dict:
    by_mllm: dict = {}
    matches = 0
    mismatches = 0
    errors = 0

    for r in results:
        if r.mllm not in by_mllm:
            by_mllm[r.mllm] = {
                "total": 0,
                "matches": 0,
                "mismatches": 0,
                "errors": 0,
                "latencies": [],
                "tokens": [],
                "correct": 0,
                "incorrect": 0,
                "evaluated": 0,
            }
        entry = by_mllm[r.mllm]
        entry["total"] += 1

        if r.error:
            errors += 1
            entry["errors"] += 1
        elif r.match is True:
            matches += 1
            entry["matches"] += 1
        elif r.match is False:
            mismatches += 1
            entry["mismatches"] += 1

        if r.latency_ms is not None and not r.error:
            entry["latencies"].append(r.latency_ms)
        if r.tokens_used is not None and not r.error:
            entry["tokens"].append(r.tokens_used)

        c = r.correct
        if c is not None:
            entry["evaluated"] += 1
            if c:
                entry["correct"] += 1
            else:
                entry["incorrect"] += 1

    # Compute aggregates per MLLM
    for mllm, entry in by_mllm.items():
        lats = entry.pop("latencies")
        toks = entry.pop("tokens")
        entry["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else None
        entry["avg_tokens"] = round(sum(toks) / len(toks), 1) if toks else None
        entry["total_tokens"] = sum(toks) if toks else None

        tok_per_sec_values = []
        for r in results:
            if r.mllm == mllm and r.tokens_used and r.latency_ms and not r.error:
                tok_per_sec_values.append(r.tokens_used / (r.latency_ms / 1000))
        entry["avg_tokens_per_sec"] = (
            round(sum(tok_per_sec_values) / len(tok_per_sec_values), 1)
            if tok_per_sec_values
            else None
        )

        ev = entry["evaluated"]
        entry["accuracy"] = round(entry["correct"] / ev, 4) if ev > 0 else None

    return {
        "total": len(results),
        "matches": matches,
        "mismatches": mismatches,
        "errors": errors,
        "by_mllm": by_mllm,
    }


def save_run(
    results: list[AnalysisResult],
    element: str,
    mllms: list[str],
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> RunRecord:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}"

    pair_ids = list({r.pair_id for r in results})
    pair_count = len(pair_ids)
    name = f"{element} · {len(mllms)} MLLM{'s' if len(mllms) != 1 else ''} · {pair_count} Paare"

    summary = _compute_summary(results)

    record = RunRecord(
        run_id=run_id,
        name=name,
        timestamp=now.isoformat(),
        element=element,
        mllms=mllms,
        pair_count=pair_count,
        total_calls=len(results),
        summary=summary,
        results=[_result_to_dict(r) for r in results],
    )

    filepath = runs_dir / f"{run_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, ensure_ascii=False, indent=2)

    return record


def load_run(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Optional[RunRecord]:
    filepath = Path(runs_dir) / f"{run_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return RunRecord(**data)


def list_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[RunRecord]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    records = []
    for filepath in sorted(runs_dir.glob("run_*.json"), reverse=True):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            records.append(RunRecord(**data))
        except Exception:
            continue
    return records


def delete_run(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> bool:
    filepath = Path(runs_dir) / f"{run_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def run_record_to_results(record: RunRecord) -> list[AnalysisResult]:
    return [_dict_to_result(d) for d in record.results]
