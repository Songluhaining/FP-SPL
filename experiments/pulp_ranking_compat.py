"""Run the repository's 50-feature experiment on one staged buggy version.

This script adapts the ranking function's return shape without editing PULP.
"""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2] / "PULP"
sys.dont_write_bytecode = True  # Keep imports from creating files in the reference repository.
sys.path.insert(0, str(REPO))

from consistent_testing_manager import FPMatricsCaculation as features
from spectrum_manager.Spectrum_Expression import (
    JACCARD, OP2, TARANTULA, BARINEL, OCHIAI,
    RUSSELL_RAO, DSTAR, GP02, GP03, GP19,
)


def ranked_scores(project_dir, failing_variants):
    search_spaces = features.get_suspicious_space_consistent_version(
        project_dir, failing_variants, 0.0, ""
    )
    all_variants = features.list_dir(features.get_variants_dir(project_dir))
    other_variants = [v for v in all_variants if v not in failing_variants]
    statement_info, passed_tests, failed_tests = features.get_infor_for_sbfl(
        project_dir, failing_variants, other_variants, "", 0.0
    )
    expressions = [
        JACCARD, OP2, TARANTULA, BARINEL, OCHIAI,
        RUSSELL_RAO, DSTAR, GP02, GP03, GP19,
    ]
    rankings = features.sbfl_ranking(
        statement_info, failed_tests, passed_tests,
        features.get_set_of_stms(search_spaces), expressions
    )
    return [
        {statement: score for statement, score, _ in rankings[expression]}
        for expression in expressions
    ]


features.ranking_suspicious_stmts = ranked_scores

if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2] / "pulp_trial" / "dataset" / "_MultipleBugs_.NOB_1.ID_1"
    features.calculate_attributes3(
        str(project), "variant_labels.csv", "attributes_temp.csv", "attributes.csv"
    )
