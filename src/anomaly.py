"""
anomaly.py
==========
Phase 4, Part 6 — Unusual (low-frequency) mobility pattern analysis.

TERMINOLOGY
-----------
This module deliberately calls its output "unusual mobility patterns",
never "incidents," "emergencies," or "anomalies" in the safety-critical
sense — those words imply a known cause (an accident, an event) that
this project has no data to support. All this module can say is: "this
specific cell-to-cell transition was observed only rarely in the sample,
relative to how often other transitions were observed."

METHOD
------
Rather than treating "not in the frequent-pattern set" as unusual (which
would flag the vast majority of legitimate but simply uncommon
transitions — not a meaningful signal), this module:

1. Builds a frequency table of every consecutive CELL-TO-CELL TRANSITION
   observed in the (compressed) taxi-day sequences — i.e. every (cell_i,
   cell_{i+1}) pair, recording both how many times it occurs in total
   (`transition_count`) and, separately, in how many DISTINCT taxi-days
   it occurs at least once (`n_taxi_days`).
2. Flags transitions as "rare mobility transitions" using
   `identify_unusual_transitions_by_occurrence()` — the PRIMARY rule:
   a transition observed in at most `config.RARE_TRANSITION_MAX_TAXI_DAYS`
   distinct taxi-days. This is a plain, defensible statement ("this route
   appeared on 2 or fewer taxi-days out of N in the whole sample") rather
   than a statistical artifact.
3. Also provides `identify_unusual_transitions_by_percentile()` — a
   supplementary, percentile-based view retained for sensitivity
   reporting only. On real taxi mobility data, many distinct transitions
   occur exactly once, so a naive percentile cutoff can flag a much
   larger fraction of distinct transitions than the percentile number
   itself would suggest (ties bunched at the bottom of the count
   distribution). Any use of this function must report the percentile
   used, the resulting count threshold, and the number/percentage of
   transitions flagged, precisely because that percentage can look
   surprisingly high and needs the reader to see why.
4. Reports which taxis/timestamps exhibited a rare transition, so a
   researcher can go look at the specific occurrence if they want to.

This says nothing about WHY a transition is rare (equipment issue,
one-off trip, genuinely novel route, data noise) — only that it is rare
in the observed sample. Findings here are described as "rare mobility
transitions" / "unusual mobility behavior", never as fraud, accidents,
incidents, or dangerous routes — the dataset does not support those
claims, and this analysis is exploratory/descriptive, not a safety-
critical anomaly detector.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from config import RARE_TRANSITION_MAX_TAXI_DAYS, UNUSUAL_TRANSITION_PERCENTILE


def build_transition_frequency_table(enriched_sequences: dict[str, list[dict]]) -> pd.DataFrame:
    """Count every consecutive (cell_i -> cell_{i+1}) transition across all
    taxi-day enriched sequences.

    Returns
    -------
    pd.DataFrame
        Columns:
            from_cell, to_cell,
            transition_count          — total occurrences across all taxi-days
                                         (a single taxi-day can contribute more
                                         than one, if the transition recurs
                                         non-consecutively within that day)
            n_taxi_days               — number of DISTINCT taxi-day sequences
                                         in which the transition occurs at
                                         least once (used by the primary,
                                         occurrence-based rare-transition rule)
            pct_of_all_transitions    — transition_count's share of the total
                                         occurrence count, for context
        Sorted by transition_count ascending (rarest first).
    """
    counter: Counter = Counter()
    taxi_day_sets: dict[tuple[str, str], set[str]] = {}

    for seq_key, seq in enriched_sequences.items():
        for i in range(len(seq) - 1):
            pair = (seq[i]["cell_id"], seq[i + 1]["cell_id"])
            counter[pair] += 1
            taxi_day_sets.setdefault(pair, set()).add(seq_key)

    if not counter:
        return pd.DataFrame(
            columns=["from_cell", "to_cell", "transition_count", "n_taxi_days", "pct_of_all_transitions"]
        )

    total = sum(counter.values())
    rows = [
        {
            "from_cell": a,
            "to_cell": b,
            "transition_count": c,
            "n_taxi_days": len(taxi_day_sets[(a, b)]),
            "pct_of_all_transitions": 100 * c / total,
        }
        for (a, b), c in counter.items()
    ]
    return pd.DataFrame(rows).sort_values("transition_count", ascending=True).reset_index(drop=True)


def identify_unusual_transitions_by_occurrence(
    transition_table: pd.DataFrame,
    max_taxi_days: int = RARE_TRANSITION_MAX_TAXI_DAYS,
) -> pd.DataFrame:
    """PRIMARY rare-transition rule: flag transitions observed in at most
    `max_taxi_days` distinct taxi-day sequences.

    This is the module's default/recommended definition — plain and
    directly interpretable ("seen on 2 or fewer taxi-days in the sample"),
    unlike a percentile cut which can flag a surprising share of distinct
    transitions when many are tied at count 1.

    Parameters
    ----------
    transition_table : pd.DataFrame
        Output of build_transition_frequency_table() (needs `n_taxi_days`).
    max_taxi_days : int
        Occurrence threshold — transitions in this many or fewer distinct
        taxi-days are flagged as rare.

    Returns
    -------
    pd.DataFrame
        Subset of transition_table flagged as rare, plus a
        `rule` column recording which rule/threshold produced the flag
        (for transparency when combined with other reporting).
    """
    if len(transition_table) == 0:
        return transition_table.assign(rule=pd.Series(dtype=str))

    flagged = transition_table[transition_table["n_taxi_days"] <= max_taxi_days].copy()
    flagged["rule"] = f"n_taxi_days <= {max_taxi_days}"
    return flagged.reset_index(drop=True)


def identify_unusual_transitions_by_percentile(
    transition_table: pd.DataFrame,
    percentile: float = UNUSUAL_TRANSITION_PERCENTILE,
) -> pd.DataFrame:
    """SUPPLEMENTARY rare-transition rule, kept for sensitivity reporting
    only: flag transitions at or below the given percentile of the
    observed transition_count distribution.

    Any use of this function's output must also report, alongside it:
    the percentile used, the resulting count threshold, and the number
    and percentage of distinct transitions flagged — because on skewed,
    tie-heavy count distributions (many transitions occurring exactly
    once) this rule can flag a much larger share of transitions than the
    percentile number alone would suggest, and that needs to be visible
    to the reader rather than asserted as a clean "top N%" result.

    Parameters
    ----------
    transition_table : pd.DataFrame
        Output of build_transition_frequency_table().
    percentile : float
        E.g. 5.0 means "the rarest 5% of observed transition counts".

    Returns
    -------
    pd.DataFrame
        The subset of transition_table flagged as unusual, plus a
        `threshold_count` column recording the cutoff used (for
        transparency in reporting).
    """
    if len(transition_table) == 0:
        return transition_table.assign(threshold_count=np.nan)

    threshold = float(np.percentile(transition_table["transition_count"], percentile))
    unusual = transition_table[transition_table["transition_count"] <= threshold].copy()
    unusual["threshold_count"] = threshold
    return unusual.reset_index(drop=True)


# Backward-compatible alias: earlier notebooks/callers used this name for
# the percentile-based rule. Kept so existing calls don't break, but new
# code should call identify_unusual_transitions_by_occurrence() (primary)
# or identify_unusual_transitions_by_percentile() (supplementary) explicitly.
identify_unusual_transitions = identify_unusual_transitions_by_percentile


def find_unusual_occurrences(
    enriched_sequences: dict[str, list[dict]],
    unusual_transitions: pd.DataFrame,
) -> pd.DataFrame:
    """Locate every actual occurrence of an unusual transition, with the
    taxi ID, timestamp, and time segment — so a researcher can inspect
    specific instances if desired.

    Returns
    -------
    pd.DataFrame
        Columns: taxi_id, from_cell, to_cell, timestamp, time_segment, day_type.
    """
    unusual_pairs = set(zip(unusual_transitions["from_cell"], unusual_transitions["to_cell"]))
    if not unusual_pairs:
        return pd.DataFrame(columns=["taxi_id", "from_cell", "to_cell", "timestamp", "time_segment", "day_type"])

    rows = []
    for seq in enriched_sequences.values():
        for i in range(len(seq) - 1):
            pair = (seq[i]["cell_id"], seq[i + 1]["cell_id"])
            if pair in unusual_pairs:
                # taxi_id comes from the record itself, not the dict key,
                # since keys are taxi-day sequence keys ("taxi_id::date")
                # when enriched_sequences was built with split_by_date=True.
                rows.append(
                    {
                        "taxi_id": seq[i + 1]["taxi_id"],
                        "from_cell": pair[0],
                        "to_cell": pair[1],
                        "timestamp": seq[i + 1]["timestamp"],
                        "time_segment": seq[i + 1]["time_segment"],
                        "day_type": seq[i + 1]["day_type"],
                    }
                )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data): most taxis take
    # A->B->C repeatedly; one taxi takes a one-off A->D transition.
    # Keys mirror pattern_ranking.build_enriched_sequences()'s taxi-day
    # format ("taxi_id::date"); each record also carries taxi_id, exactly
    # as build_enriched_sequences() produces it.
    fixture_sequences = {
        "1::2008-02-02": [{"taxi_id": "1", "cell_id": "A", "timestamp": pd.Timestamp("2008-02-02 08:00"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "1", "cell_id": "B", "timestamp": pd.Timestamp("2008-02-02 08:05"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "1", "cell_id": "C", "timestamp": pd.Timestamp("2008-02-02 08:10"), "time_segment": "morning", "day_type": "weekday"}],
        "2::2008-02-02": [{"taxi_id": "2", "cell_id": "A", "timestamp": pd.Timestamp("2008-02-02 09:00"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "2", "cell_id": "B", "timestamp": pd.Timestamp("2008-02-02 09:05"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "2", "cell_id": "C", "timestamp": pd.Timestamp("2008-02-02 09:10"), "time_segment": "morning", "day_type": "weekday"}],
        "3::2008-02-02": [{"taxi_id": "3", "cell_id": "A", "timestamp": pd.Timestamp("2008-02-02 10:00"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "3", "cell_id": "B", "timestamp": pd.Timestamp("2008-02-02 10:05"), "time_segment": "morning", "day_type": "weekday"},
              {"taxi_id": "3", "cell_id": "C", "timestamp": pd.Timestamp("2008-02-02 10:10"), "time_segment": "morning", "day_type": "weekday"}],
        "4::2008-02-02": [{"taxi_id": "4", "cell_id": "A", "timestamp": pd.Timestamp("2008-02-02 23:00"), "time_segment": "night", "day_type": "weekday"},
              {"taxi_id": "4", "cell_id": "D", "timestamp": pd.Timestamp("2008-02-02 23:05"), "time_segment": "night", "day_type": "weekday"}],
    }
    print("Building transition frequency table from a synthetic TEST FIXTURE (not real data):\n")
    freq_table = build_transition_frequency_table(fixture_sequences)
    print(freq_table)

    print("\nPrimary rule — rare by occurrence (<=1 taxi-day for this tiny fixture):")
    rare_by_occurrence = identify_unusual_transitions_by_occurrence(freq_table, max_taxi_days=1)
    print(rare_by_occurrence)

    print("\nSupplementary rule — rare by percentile (30th pct for this tiny fixture):")
    rare_by_percentile = identify_unusual_transitions_by_percentile(freq_table, percentile=30.0)
    print(rare_by_percentile)

    print("\nOccurrences of the primary rule's flagged transitions:")
    print(find_unusual_occurrences(fixture_sequences, rare_by_occurrence))
