"""Plain-data description of the AION-2 parity cases.

Shared by ``generate_aion_parity.py`` (run in an AION-2 environment) and
``tests/test_aion_parity.py``; it deliberately imports neither package.
"""

SURVEYS = ("a", "b", "c")
RADIUS_ARCSEC = 1.0
DEDUPE_RADIUS_ARCSEC = {"a": 0.5, "b": 0.4, "c": 0.0}
PAIR_RADIUS_OVERRIDE = (("a", "c"), 0.8)

# label -> (mode kind, keyword arguments shared by both implementations)
CASES = {
    "degenerate_ab_mutual_nearest": (
        "degenerate",
        {"surveys": ["a", "b"], "policy": "mutual_nearest"},
    ),
    "degenerate_ab_mutual_unique": (
        "degenerate",
        {"surveys": ["a", "b"], "policy": "mutual_unique"},
    ),
    "degenerate_ab_anchored_nearest": (
        "degenerate",
        {"surveys": ["a", "b"], "policy": "anchored_nearest"},
    ),
    "degenerate_ab_anchored_unique": (
        "degenerate",
        {"surveys": ["a", "b"], "policy": "anchored_unique"},
    ),
    "degenerate_abc_mutual_nearest": (
        "degenerate",
        {"surveys": ["a", "b", "c"], "policy": "mutual_nearest"},
    ),
    "degenerate_abc_mutual_unique": (
        "degenerate",
        {"surveys": ["a", "b", "c"], "policy": "mutual_unique"},
    ),
    "entitywise_refuse_singleton": (
        "entitywise",
        {"resolver": "refuse", "disputed": "singleton"},
    ),
    "entitywise_refuse_drop": (
        "entitywise",
        {"resolver": "refuse", "disputed": "drop"},
    ),
    "entitywise_sequential_singleton": (
        "entitywise",
        {
            "resolver": "sequential",
            "disputed": "singleton",
            "priority": ["b", "a", "c"],
        },
    ),
    "entitywise_sequential_drop": (
        "entitywise",
        {"resolver": "sequential", "disputed": "drop", "priority": ["b", "a", "c"]},
    ),
    "entitywise_split_singleton": (
        "entitywise",
        {"resolver": "split", "disputed": "singleton"},
    ),
    "entitywise_split_drop": ("entitywise", {"resolver": "split", "disputed": "drop"}),
    "entitywise_split_size_cap_3": ("entitywise", {"resolver": "split", "size_cap": 3}),
    "entitywise_refuse_selection": (
        "entitywise",
        {"selection": {"min_surveys": 2, "must_include_surveys": ["a"]}},
    ),
}
