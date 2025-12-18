import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List


def _key_from_path_last2parts(p: str) -> str:
    """Return a stable matching key based on last 2 path parts: parent_dir/filename.

    Example:
        /a/b/Annual/1318498.mp4 -> annual/1318498.mp4
    """
    # Normalize separators and remove trailing slashes
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[-2].lower()}/{parts[-1].lower()}"


def build_index_last2parts(root_dir: str, exts: List[str]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    root = Path(root_dir)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = Path(fn).suffix.lower()
            if ext not in exts:
                continue
            fp = str(Path(dp) / fn)
            key = _key_from_path_last2parts(fp)
            index.setdefault(key, []).append(fp)

    # sort for deterministic selection
    for k in index:
        index[k].sort()
    return index


def filter_csv(
    input_csv: str,
    output_csv: str,
    search_root: str,
    exts: List[str],
    match_column: str = "video_path",
    replace_column: str = "video_path",
    dedup_by_resolved_path: bool = True,
):
    suffix_index = build_index_last2parts(search_root, exts)

    total = 0
    kept = 0
    missing = 0
    ambiguous = 0
    dedup_skipped = 0
    seen_resolved = set()

    with open(input_csv, "r", encoding="utf-8") as fin, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {input_csv}")

        fieldnames = list(reader.fieldnames)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            raw_path = row.get(match_column, "")
            key = _key_from_path_last2parts(raw_path)

            cands = suffix_index.get(key, [])
            if not cands:
                missing += 1
                continue

            if len(cands) > 1:
                ambiguous += 1
                # Don't guess: skip ambiguous matches to avoid incorrect mapping
                continue

            resolved = cands[0]
            if dedup_by_resolved_path:
                if resolved in seen_resolved:
                    dedup_skipped += 1
                    continue
                seen_resolved.add(resolved)

            row[replace_column] = resolved
            writer.writerow(row)
            kept += 1

    return {
        "total": total,
        "kept": kept,
        "missing": missing,
        "ambiguous": ambiguous,
        "dedup_skipped": dedup_skipped,
        "index_keys": len(suffix_index),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter train_data.csv by matching last 2 path parts (parent_dir/filename) to files under wanErase"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV (e.g. .../train_data.csv)",
    )
    parser.add_argument(
        "--search_root",
        type=str,
        required=True,
        help="Search root (e.g. .../data/videoEdit/wanErase)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".mp4",
        help="Comma-separated extensions to search, default: .mp4",
    )
    parser.add_argument(
        "--match_column",
        type=str,
        default="video_path",
        help="Which CSV column to match from, default: video_path",
    )
    parser.add_argument(
        "--replace_column",
        type=str,
        default="video_path",
        help="Which CSV column to write matched path into, default: video_path",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable deduplication by resolved local path (default is to dedup)",
    )

    args = parser.parse_args()
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]

    stats = filter_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        search_root=args.search_root,
        exts=exts,
        match_column=args.match_column,
        replace_column=args.replace_column,
        dedup_by_resolved_path=(not args.no_dedup),
    )

    print("Done")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
