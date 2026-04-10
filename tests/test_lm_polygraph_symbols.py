"""Canary tests for lm_polygraph symbol stability.

thinkbooster imports many symbols from ``lm_polygraph``, often lazily
(inside functions) or guarded by ``try/except ImportError``. When
lm_polygraph is upgraded, a renamed or removed symbol will silently
disable the affected code path without breaking any other test — this is
exactly how the ``VLLMLogprobsCalculator`` -> ``VLLMLogprobsExtractionCalculator``
rename in lm_polygraph 0.6.0 slipped through CI.

This module lists every ``lm_polygraph`` symbol that thinkbooster depends
on. If any rename or removal happens upstream, a targeted test case fails
loudly and points at the source file that uses it.

When adding a new ``from lm_polygraph import ...`` anywhere in the
codebase, add a matching entry to ``SYMBOLS`` below.
"""

from __future__ import annotations

import importlib

import pytest

# (module, attribute, used_by) — used_by appears in the failure message
# so the root cause is obvious from CI output alone.
SYMBOLS: list[tuple[str, str, str]] = [
    # thinkbooster/generators/api.py
    (
        "lm_polygraph.utils.api_with_uncertainty",
        "APIWithUncertainty",
        "thinkbooster/generators/api.py",
    ),
    # thinkbooster/generators/huggingface.py, scripts/run_tts_eval.py
    ("lm_polygraph", "WhiteboxModel", "thinkbooster/generators/huggingface.py"),
    # thinkbooster/generators/vllm.py, service_app/core/strategy_manager.py
    ("lm_polygraph.utils", "VLLMWithUncertainty", "thinkbooster/generators/vllm.py"),
    # thinkbooster/models/blackboxmodel_with_streaming.py,
    # thinkbooster/strategies/deepconf/strategy.py
    (
        "lm_polygraph",
        "BlackboxModel",
        "thinkbooster/models/blackboxmodel_with_streaming.py",
    ),
    (
        "lm_polygraph.utils.generation_parameters",
        "GenerationParameters",
        "thinkbooster/models/blackboxmodel_with_streaming.py",
    ),
    # thinkbooster/strategies/deepconf/utils.py
    (
        "lm_polygraph.estimators",
        "MaximumTokenProbability",
        "thinkbooster/strategies/deepconf/utils.py",
    ),
    (
        "lm_polygraph.utils.token_restoration",
        "Categorical",
        "thinkbooster/strategies/deepconf/utils.py",
    ),
    # thinkbooster/scorers/multi_scorer.py,
    # scripts/run_tts_eval.py, service_app/core/strategy_manager.py
    (
        "lm_polygraph.stat_calculators",
        "VLLMLogprobsExtractionCalculator",
        "thinkbooster/scorers/multi_scorer.py",
    ),
    (
        "lm_polygraph.stat_calculators",
        "EntropyCalculator",
        "thinkbooster/scorers/multi_scorer.py",
    ),
    # thinkbooster/scorers/estimator_uncertainty_pd.py
    (
        "lm_polygraph.estimators.estimator",
        "Estimator",
        "thinkbooster/scorers/estimator_uncertainty_pd.py",
    ),
    # thinkbooster/scorers/step_scorer_prm.py
    (
        "lm_polygraph.stat_calculators",
        "StatCalculator",
        "thinkbooster/scorers/step_scorer_prm.py",
    ),
    (
        "lm_polygraph.stat_calculators.extract_claims",
        "Claim",
        "thinkbooster/scorers/step_scorer_prm.py",
    ),
    # scripts/run_tts_eval.py, service_app/core/strategy_manager.py
    (
        "lm_polygraph.estimators",
        "MaximumSequenceProbability",
        "scripts/run_tts_eval.py",
    ),
    ("lm_polygraph.estimators", "MeanTokenEntropy", "scripts/run_tts_eval.py"),
    ("lm_polygraph.estimators", "Perplexity", "scripts/run_tts_eval.py"),
    ("lm_polygraph.utils", "APIWithUncertainty", "scripts/run_tts_eval.py"),
    (
        "lm_polygraph.model_adapters",
        "WhiteboxModelvLLM",
        "scripts/run_tts_eval.py",
    ),
    # thinkbooster/evaluation/alignscore.py (optional — alignscore extra)
    (
        "lm_polygraph.generation_metrics.alignscore_utils",
        "AlignScorer",
        "thinkbooster/evaluation/alignscore.py",
    ),
    (
        "lm_polygraph.generation_metrics.generation_metric",
        "GenerationMetric",
        "thinkbooster/evaluation/alignscore.py",
    ),
]


@pytest.mark.parametrize(
    "module_path,attribute,used_by",
    SYMBOLS,
    ids=[f"{m}.{a}" for m, a, _ in SYMBOLS],
)
def test_lm_polygraph_symbol_exists(
    module_path: str, attribute: str, used_by: str
) -> None:
    """Every lm_polygraph symbol used in thinkbooster must resolve.

    If the parent module itself cannot be imported (e.g. optional extra
    not installed), the test is skipped — we only want to fail on renames
    and removals, not on missing optional dependencies.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(
            f"{module_path} is unavailable ({exc}); "
            f"skipping — likely an optional dependency of lm_polygraph."
        )
    assert hasattr(module, attribute), (
        f"{module_path}.{attribute} is missing. "
        f"Used by {used_by}. lm_polygraph may have renamed or removed it; "
        f"update the call site and this canary together."
    )


def test_multi_scorer_lazy_imports_resolve() -> None:
    """Exercise the lazy imports in thinkbooster.scorers.multi_scorer.

    The calculator factories import ``lm_polygraph`` symbols inside the
    function body. Without this test, a rename in lm_polygraph silently
    breaks runtime scoring while every other unit test stays green.
    """
    pytest.importorskip("lm_polygraph.stat_calculators")

    from thinkbooster.scorers import multi_scorer

    # Reset module-level caches so we test the actual import path,
    # not a cached instance from an earlier test run.
    multi_scorer._calc_basic = None
    multi_scorer._calc_matrix = None
    multi_scorer._calc_entropy = None

    basic = multi_scorer._get_basic_calculator()
    matrix = multi_scorer._get_matrix_calculator()
    entropy = multi_scorer._get_entropy_calculator()

    assert type(basic).__name__ == "VLLMLogprobsExtractionCalculator"
    assert type(matrix).__name__ == "VLLMLogprobsExtractionCalculator"
    assert type(entropy).__name__ == "EntropyCalculator"
