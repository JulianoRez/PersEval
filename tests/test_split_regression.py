"""Regression test for the split protocol.

The five ``get_splits`` implementations in ``perseval/data.py`` are being
consolidated into a single generic engine. Before that refactoring can be
trusted, we need proof that it does not move a single annotation between
splits.

This script records, for every dataset and every valid combination of flags,
exactly which users, texts and ``(user, text)`` annotations end up in the
training, adaptation and test splits, and reduces each of those to a digest.
Re-running it after a change reports any split whose contents moved.

Record the baseline BEFORE refactoring::

    python tests/test_split_regression.py --update

Check nothing moved AFTER refactoring::

    python tests/test_split_regression.py

Options::

    --datasets EPIC,BREXIT     restrict to a subset (default: all but MHS)
    --datasets all             include MHS, which is slow (~8k annotators)
"""

import os

# tqdm must be silenced before perseval.data imports it.
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import contextlib
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perseval.data import Epic, Brexit, DICES, MHS, MD  # noqa: E402

BASELINE_PATH = Path(__file__).parent / "baselines" / "splits.json"

# Each dataset is rebuilt from scratch for every combination of flags, because
# ``get_splits`` mutates the instance it is called on.
DATASET_FACTORIES = {
    "EPIC": lambda: Epic("irony"),
    "BREXIT": lambda: Brexit(),
    "DICES": lambda: DICES("Q2_harmful_content_overall"),
    "MD": lambda: MD("offensiveness"),
    "MHS": lambda: MHS("hateful"),
}

# MHS has ~7.9k annotators; recording it takes long enough that it is opt-in.
DEFAULT_DATASETS = ["EPIC", "BREXIT", "DICES", "MD"]

SPLIT_NAMES = ("train", "adaptation", "test")


def flag_combinations(dataset_name):
    """Yield every combination of flags the library accepts for this dataset."""
    for user_adaptation in (False, "train", "test"):
        for extended in (False, True):
            for named in (False, True):
                # Documented as invalid: with no adaptation and no traits, the
                # model knows nothing at all about the test users.
                if not user_adaptation and not named:
                    continue
                # MD-Agreement ships no annotator metadata.
                if dataset_name == "MD" and named:
                    continue
                yield user_adaptation, extended, named


def combination_key(user_adaptation, extended, named):
    return f"adaptation={user_adaptation}|extended={extended}|named={named}"


def _digest(*groups):
    """Order-independent digest of several groups of strings."""
    sha = hashlib.sha256()
    for group in groups:
        for item in group:
            sha.update(item.encode("utf-8"))
            sha.update(b"\x1e")
        sha.update(b"\x1d")
    return sha.hexdigest()[:16]


def _trait_rows(split):
    """Flatten every user's traits into sortable ``user|dimension|value`` rows."""
    rows = []
    for user_id, user in split.users.items():
        for dimension, values in user.traits.items():
            rows.append(f"{user_id}\x1f{dimension}\x1f{values[0]}")
    return sorted(rows)


def fingerprint(dataset):
    """Reduce the three splits of a dataset to counts plus a digest each."""
    summary = {}
    for name in SPLIT_NAMES:
        split = getattr(dataset, f"{'training' if name == 'train' else name}_set")
        users = sorted(str(u) for u in split.users)
        texts = sorted(str(t) for t in split.texts)
        annotations = sorted(f"{u}\x1f{t}" for u, t in split.annotation)
        # Traits are recorded separately: ``named`` changes them without moving
        # a single annotation, and ``read_traits`` is one of the hooks the
        # refactoring introduces, so it needs its own guard.
        traits = _trait_rows(split)
        summary[name] = {
            "n_users": len(users),
            "n_texts": len(texts),
            "n_annotations": len(annotations),
            "n_traits": len(traits),
            "digest": _digest(users, texts, annotations),
            "traits_digest": _digest(traits),
        }
    return summary


@contextlib.contextmanager
def quiet():
    """Silence the progress bars and the split statistics while recording."""
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                yield
    finally:
        logging.disable(logging.NOTSET)


def record(dataset_name):
    """Fingerprint every valid flag combination for one dataset."""
    recorded = {}
    for user_adaptation, extended, named in flag_combinations(dataset_name):
        key = combination_key(user_adaptation, extended, named)
        try:
            with quiet():
                dataset = DATASET_FACTORIES[dataset_name]()
                dataset.get_splits(
                    user_adaptation=user_adaptation,
                    extended=extended,
                    named=named,
                )
            recorded[key] = fingerprint(dataset)
        except Exception as exc:
            # A combination that raises today must keep raising: that is part
            # of the behaviour the refactoring has to preserve.
            recorded[key] = {"raises": f"{type(exc).__name__}: {str(exc)[:160]}"}
        print(f"  {key}  ->  {_describe(recorded[key])}", flush=True)
    return recorded


def _describe(entry):
    if "raises" in entry:
        return entry["raises"]
    parts = [f"{name}:{entry[name]['n_annotations']}" for name in SPLIT_NAMES]
    parts.append(f"traits:{sum(entry[name]['n_traits'] for name in SPLIT_NAMES)}")
    return " ".join(parts)


def compare(baseline, current):
    """Return a list of human-readable differences between two recordings."""
    problems = []
    for dataset_name in sorted(set(baseline) | set(current)):
        if dataset_name not in baseline:
            problems.append(f"{dataset_name}: not in the baseline (run --update)")
            continue
        if dataset_name not in current:
            continue  # simply not part of this run
        old, new = baseline[dataset_name], current[dataset_name]
        for key in sorted(set(old) | set(new)):
            if key not in old:
                problems.append(f"{dataset_name} [{key}]: new combination")
            elif key not in new:
                problems.append(f"{dataset_name} [{key}]: combination disappeared")
            elif old[key] != new[key]:
                problems.append(
                    f"{dataset_name} [{key}]:\n"
                    f"      before: {_describe(old[key])}\n"
                    f"      after:  {_describe(new[key])}"
                )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="record the current behaviour as the new baseline")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="comma-separated names, or 'all'")
    args = parser.parse_args()

    names = (list(DATASET_FACTORIES) if args.datasets == "all"
             else [n.strip().upper() for n in args.datasets.split(",")])
    unknown = [n for n in names if n not in DATASET_FACTORIES]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    current = {}
    for name in names:
        print(f"\n{name}")
        current[name] = record(name)

    if args.update:
        baseline = {}
        if BASELINE_PATH.exists():
            baseline = json.loads(BASELINE_PATH.read_text())
        baseline.update(current)
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"\nBaseline written to {BASELINE_PATH.relative_to(REPO_ROOT)} "
              f"({len(baseline)} dataset(s)).")
        return 0

    if not BASELINE_PATH.exists():
        print(f"\nNo baseline at {BASELINE_PATH.relative_to(REPO_ROOT)}. "
              f"Record one first with --update.")
        return 1

    problems = compare(json.loads(BASELINE_PATH.read_text()), current)
    if problems:
        print(f"\n{len(problems)} split(s) changed:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nEvery split matches the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
