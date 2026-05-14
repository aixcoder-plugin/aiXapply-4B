#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "results" / "apply_correctness_judge"
DEFAULT_PATTERN = "*.judged.jsonl"
DEFAULT_LANGUAGE_ORDER = [
    "tsx",
    "javascript",
    "typescript",
    "css",
    "jsx",
    "rust",
    "sql",
    "vue",
]
DEFAULT_ALIASES = {
    "deepseek-v3-2": "DeepSeek3.2",
    "FastApply-7B": "fast-apply",
    "aiXapply-test": "aiXapply-rl",
    "aiXapply-sft_fastapply": "aiXapply-sft",
}


@dataclass
class ModelStats:
    label: str
    path: Path
    counts_by_language: Counter[str] = field(default_factory=Counter)
    correct_by_language: Counter[str] = field(default_factory=Counter)
    total_count: int = 0
    correct_count: int = 0

    def accuracy_for(self, language: str) -> float | None:
        total = self.counts_by_language.get(language, 0)
        if total == 0:
            return None
        return self.correct_by_language.get(language, 0) / total

    @property
    def overall_accuracy(self) -> float | None:
        if self.total_count == 0:
            return None
        return self.correct_count / self.total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize apply-correctness judge results by programming language and output a Markdown table."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing *.judged.jsonl files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern used during automatic discovery.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Explicitly specify data sources. This option can be repeated. "
            "Example: --source DeepSeek3.2=experiments/evaluation/results/"
            "apply_correctness_judge/deepseek...judged.jsonl"
        ),
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="FILE_OR_STEM=LABEL",
        help="Rename discovered file labels during automatic discovery. This option can be repeated.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGE_ORDER),
        help='Comma-separated language order. Use "all" to output every discovered language.',
    )
    parser.add_argument(
        "--column-order",
        default="",
        help="Comma-separated column order. Missing sources are allowed and will produce empty cells.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. Defaults to stdout.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def parse_mapping_entries(entries: Iterable[str], entry_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid {entry_name} entry {entry!r}; expected KEY=VALUE.")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid {entry_name} entry {entry!r}; key/value cannot be empty.")
        mapping[key] = value
    return mapping


def normalize_language(language: object) -> str:
    value = str(language or "unknown").strip().lower()
    return value or "unknown"


def discover_sources(
    input_dir: Path,
    pattern: str,
    aliases: dict[str, str],
) -> dict[str, Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    sources: dict[str, Path] = {}
    for path in sorted(input_dir.glob(pattern)):
        raw_stem = path.stem.removesuffix(".judged")
        candidates = [
            path.name,
            path.stem,
            raw_stem,
            path.name.removesuffix(".judged.jsonl"),
        ]
        label = None
        for candidate in candidates:
            label = aliases.get(candidate) or DEFAULT_ALIASES.get(candidate)
            if label is not None:
                break
        if label is None:
            for mapping in (aliases, DEFAULT_ALIASES):
                for alias_key, alias_label in mapping.items():
                    if any(candidate.startswith(alias_key) for candidate in candidates):
                        label = alias_label
                        break
                if label is not None:
                    break
        if label is None:
            label = raw_stem
        if label in sources:
            raise ValueError(f"Duplicate label {label!r} discovered for {path} and {sources[label]}")
        sources[label] = path.resolve()
    return sources


def load_sources(args: argparse.Namespace) -> dict[str, Path]:
    if args.source:
        raw_sources = parse_mapping_entries(args.source, "source")
        return {label: resolve_repo_path(path_str) for label, path_str in raw_sources.items()}

    alias_overrides = parse_mapping_entries(args.alias, "alias")
    input_dir = resolve_repo_path(args.input_dir)
    return discover_sources(input_dir=input_dir, pattern=args.pattern, aliases=alias_overrides)


def compute_stats(label: str, path: Path) -> ModelStats:
    if not path.exists():
        raise FileNotFoundError(f"Judged file does not exist: {path}")

    stats = ModelStats(label=label, path=path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc

            language = normalize_language(record.get("language"))
            verdict = str(record.get("judge_result") or "").strip().lower()

            stats.total_count += 1
            stats.counts_by_language[language] += 1
            if verdict == "correct":
                stats.correct_count += 1
                stats.correct_by_language[language] += 1
    return stats


def select_languages(stats_by_label: dict[str, ModelStats], languages_arg: str) -> list[str]:
    discovered_languages = {
        language
        for stats in stats_by_label.values()
        for language, count in stats.counts_by_language.items()
        if count > 0
    }
    if languages_arg.strip().lower() == "all":
        preferred = [lang for lang in DEFAULT_LANGUAGE_ORDER if lang in discovered_languages]
        remainder = sorted(discovered_languages - set(preferred))
        return preferred + remainder

    languages = [normalize_language(item) for item in languages_arg.split(",") if item.strip()]
    return languages


def select_column_order(stats_by_label: dict[str, ModelStats], column_order_arg: str) -> list[str]:
    if column_order_arg.strip():
        return [item.strip() for item in column_order_arg.split(",") if item.strip()]
    return list(stats_by_label.keys())


def warn_on_count_mismatch(stats_by_label: dict[str, ModelStats], languages: Iterable[str]) -> None:
    for language in languages:
        counts = {
            label: stats.counts_by_language.get(language, 0)
            for label, stats in stats_by_label.items()
            if stats.counts_by_language.get(language, 0) > 0
        }
        if len(set(counts.values())) > 1:
            details = ", ".join(f"{label}={count}" for label, count in counts.items())
            print(
                f"[WARN] inconsistent sample counts for language {language!r}: {details}",
                file=sys.stderr,
            )


def format_accuracy(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def format_language_cell(
    language: str,
    column_order: Iterable[str],
    stats_by_label: dict[str, ModelStats],
) -> str:
    counts = []
    for label in column_order:
        stats = stats_by_label.get(label)
        if not stats:
            continue
        count = stats.counts_by_language.get(language, 0)
        if count > 0:
            counts.append((label, count))

    if not counts:
        return language

    distinct_counts = {count for _, count in counts}
    if len(distinct_counts) == 1:
        return f"{language} ({counts[0][1]})"

    count_summary = ", ".join(f"{label}:{count}" for label, count in counts)
    return f"{language} ({count_summary})"


def build_markdown_table(
    stats_by_label: dict[str, ModelStats],
    languages: list[str],
    column_order: list[str],
) -> str:
    headers = ["Language"] + column_order
    separator = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for language in languages:
        row = [format_language_cell(language, column_order, stats_by_label)]
        for label in column_order:
            stats = stats_by_label.get(label)
            row.append(format_accuracy(stats.accuracy_for(language)) if stats else "")
        rows.append("| " + " | ".join(row) + " |")

    average_row = ["Average"]
    for label in column_order:
        stats = stats_by_label.get(label)
        average_row.append(format_accuracy(stats.overall_accuracy) if stats else "")
    rows.append("| " + " | ".join(average_row) + " |")
    return "\n".join(rows)


def write_output(table: str, output_path_str: str) -> None:
    if not output_path_str:
        print(table)
        return

    output_path = resolve_repo_path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[OK] wrote {output_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    sources = load_sources(args)
    if not sources:
        raise FileNotFoundError("No judged jsonl files were found. Please check --input-dir/--pattern.")

    stats_by_label = {label: compute_stats(label, path) for label, path in sources.items()}
    languages = select_languages(stats_by_label, args.languages)
    column_order = select_column_order(stats_by_label, args.column_order)
    warn_on_count_mismatch(stats_by_label, languages)

    table = build_markdown_table(stats_by_label, languages, column_order)
    write_output(table, args.output)


if __name__ == "__main__":
    main()
