import logging
import os

import xml.etree.ElementTree as ET

from util.FileManager import get_variant_dir, get_test_coverage_dir, join_path, \
    get_failing_variants, get_spectrum_failed_coverage_file_name_with_version, \
    get_spectrum_passed_coverage_file_name_with_version, list_dir, get_variants_dir
from ranking.RankingManager import get_set_of_stms

from ranking.Spectrum_Expression import *
from ranking.Keywords import *
from util.TestingCoverageManager import statement_coverage_of_variants


def features_ranking_multiple_bugs(buggy_statements, mutated_project_dir, search_spaces, filter_coverage_rate, spectrum_expressions,
                                   spectrum_coverage_prefix=""):
    statements_sets = {}
    statements_sets[SS_STMS_IN_F_PRODUCTS] = get_set_of_stms(search_spaces[SS_STMS_IN_F_PRODUCTS])
    statements_sets[SS_SLICING] = get_set_of_stms(search_spaces[SS_SLICING])
    total_variants = 0
    variants_testing_coverage = {}
    if filter_coverage_rate > 0:
        variants_testing_coverage = statement_coverage_of_variants(mutated_project_dir, spectrum_coverage_prefix)
    failing_variants = get_failing_variants(mutated_project_dir)
    features_info = {SS_STMS_IN_F_PRODUCTS: {}, SS_SLICING: {}}

    variants_dir = get_variants_dir(mutated_project_dir)
    for variant in list_dir(variants_dir):
        if variant in failing_variants or (filter_coverage_rate == 0 or (variants_testing_coverage[variant] >= filter_coverage_rate)):
            total_variants += 1
            variant_dir = get_variant_dir(mutated_project_dir, variant)
            features_info = get_coverage_infor_of_variants(variant, variant_dir, failing_variants, features_info, statements_sets,
                                                           spectrum_coverage_prefix)

    total_passes = total_variants - len(failing_variants)
    total_fails = len(failing_variants)

    # there are no passing variants with test coverage > threshold
    if (total_passes == 0):
        return -2, -2, -2
    # there are no failing variants
    if (total_fails == 0):
        return -2, -2, -2
    feature_based_rank = {}
    for spectrum_expression in spectrum_expressions:
        features_info[SS_STMS_IN_F_PRODUCTS] = features_suspiciousness_calculation(features_info[SS_STMS_IN_F_PRODUCTS], total_passes, total_fails, spectrum_expression)

        feature_based_rank[spectrum_expression] = {FB_RANK:{}, FB_TC_RANK:{}}
        for stm in buggy_statements:
            feature_rank, stm_rank = search_rank_worst_case(stm, features_info[SS_STMS_IN_F_PRODUCTS], features_info[SS_STMS_IN_F_PRODUCTS], spectrum_expression)
            feature_based_rank[spectrum_expression][FB_RANK][stm] = {}
            feature_based_rank[spectrum_expression][FB_RANK][stm][RANK] = stm_rank
            feature_based_rank[spectrum_expression][FB_RANK][stm][EXAM] = (stm_rank/ len(statements_sets[SS_STMS_IN_F_PRODUCTS])) * 100

        for stm in buggy_statements:
            if(stm in statements_sets[SS_SLICING]):
                feature_rank, stm_rank = search_rank_worst_case(stm, features_info[SS_STMS_IN_F_PRODUCTS], features_info[SS_SLICING], spectrum_expression)
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm] = {}
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm][RANK] = stm_rank
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm][EXAM] = (stm_rank / len(statements_sets[SS_SLICING])) * 100
            else:
                feature_based_rank[spectrum_expression][FB_TC_RANK] = feature_based_rank[spectrum_expression][FB_RANK]

    return feature_based_rank


def features_suspiciousness_calculation(features_info, total_passes, total_fails, spectrum_expression):
    for id in features_info.keys():
        if spectrum_expression == TARANTULA:
            features_info[id][TARANTULA_SCORE] = tarantula_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                       len(features_info[id][VARIANTS_PASSED]),
                                                                       total_fails,
                                                                       total_passes)
        elif spectrum_expression == OCHIAI:
            features_info[id][OCHIAI_SCORE] = ochiai_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == OP2:
            features_info[id][OP2_SCORE] = op2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                           len(features_info[id][VARIANTS_PASSED]), total_fails,
                                                           total_passes)
        elif spectrum_expression == BARINEL:
            features_info[id][BARINEL_SCORE] = barinel_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]), total_fails,
                                                                   total_passes)
        elif spectrum_expression == DSTAR:
            features_info[id][DSTAR_SCORE] = dstar_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == DSTAR:
            features_info[id][DSTAR_SCORE] = dstar_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == RUSSELL_RAO:
            features_info[id][RUSSELL_RAO_SCORE] = russell_rao_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                           len(features_info[id][VARIANTS_PASSED]),
                                                                           total_fails, total_passes)
        elif spectrum_expression == SIMPLE_MATCHING:
            features_info[id][SIMPLE_MATCHING_SCORE] = simple_matching_calculation(
                len(features_info[id][VARIANTS_FAILED]), len(features_info[id][VARIANTS_PASSED]),
                total_fails, total_passes)
        elif spectrum_expression == ROGERS_TANIMOTO:
            features_info[id][ROGERS_TANIMOTO_SCORE] = rogers_tanimoto_calculation(
                len(features_info[id][VARIANTS_FAILED]), len(features_info[id][VARIANTS_PASSED]),
                total_fails, total_passes)
        elif spectrum_expression == AMPLE:
            features_info[id][AMPLE_SCORE] = ample2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == JACCARD:
            features_info[id][JACCARD_SCORE] = jaccard_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]),
                                                                   total_fails, total_passes)
        elif spectrum_expression == COHEN:
            features_info[id][COHEN_SCORE] = cohen_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == SCOTT:
            features_info[id][SCOTT_SCORE] = scott_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == ROGOT1:
            features_info[id][ROGOT1_SCORE] = rogot1_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == GEOMETRIC_MEAN:
            features_info[id][GEOMETRIC_MEAN_SCORE] = geometric_mean_calculation(
                len(features_info[id][VARIANTS_FAILED]), len(features_info[id][VARIANTS_PASSED]),
                total_fails, total_passes)
        elif spectrum_expression == M2:
            features_info[id][M2_SCORE] = m2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                         len(features_info[id][VARIANTS_PASSED]),
                                                         total_fails, total_passes)
        elif spectrum_expression == WONG1:
            features_info[id][WONG1_SCORE] = wong1_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == SOKAL:
            features_info[id][SOKAL_SCORE] = sokal_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        # new
        elif spectrum_expression == SORENSEN_DICE:
            features_info[id][SORENSEN_DICE_SCORE] = sorensen_dice_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                               len(features_info[id][VARIANTS_PASSED]),
                                                                               total_fails, total_passes)
        elif spectrum_expression == DICE:
            features_info[id][DICE_SCORE] = dice_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                             len(features_info[id][VARIANTS_PASSED]),
                                                             total_fails, total_passes)
        elif spectrum_expression == HUMANN:
            features_info[id][HUMANN_SCORE] = humman_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == M1:
            features_info[id][M1_SCORE] = m1_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                         len(features_info[id][VARIANTS_PASSED]),
                                                         total_fails, total_passes)
        elif spectrum_expression == WONG2:
            features_info[id][WONG2_SCORE] = wong2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == WONG3:
            features_info[id][WONG3_SCORE] = wong3_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                               len(features_info[id][VARIANTS_PASSED]),
                                                               total_fails, total_passes)
        elif spectrum_expression == ZOLTAR:
            features_info[id][ZOLTAR_SCORE] = zoltar_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == OVERLAP:
            features_info[id][OVERLAP_SCORE] = overlap_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]),
                                                                   total_fails, total_passes)
        elif spectrum_expression == EUCLID:
            features_info[id][EUCLID_SCORE] = euclid_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == ROGOT2:
            features_info[id][ROGOT2_SCORE] = rogot2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == HAMMING:
            features_info[id][HAMMING_SCORE] = hamming_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]),
                                                                   total_fails, total_passes)
        elif spectrum_expression == FLEISS:
            features_info[id][FLEISS_SCORE] = fleiss_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                 len(features_info[id][VARIANTS_PASSED]),
                                                                 total_fails, total_passes)
        elif spectrum_expression == ANDERBERG:
            features_info[id][ANDERBERG_SCORE] = anderberg_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                       len(features_info[id][VARIANTS_PASSED]),
                                                                       total_fails, total_passes)
        elif spectrum_expression == GOODMAN:
            features_info[id][GOODMAN_SCORE] = goodman_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]),
                                                                   total_fails, total_passes)
        elif spectrum_expression == HARMONIC_MEAN:
            features_info[id][HARMONIC_MEAN_SCORE] = harmonic_mean_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                               len(features_info[id][VARIANTS_PASSED]),
                                                                               total_fails, total_passes)
        elif spectrum_expression == KULCZYNSKI1:
            features_info[id][KULCZYNSKI1_SCORE] = kulczynski1_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                           len(features_info[id][VARIANTS_PASSED]),
                                                                           total_fails, total_passes)
        elif spectrum_expression == KULCZYNSKI2:
            features_info[id][KULCZYNSKI2_SCORE] = kulczynski2_calculation(len(features_info[id][VARIANTS_FAILED]),
                                                                           len(features_info[id][VARIANTS_PASSED]),
                                                                           total_fails, total_passes)

    return features_info


def get_coverage_infor_of_variants(variant, variant_dir, failing_variants, features_coverage_info, search_spaces,
                                   spectrum_coverage_prefix):
    test_coverage_dir = get_test_coverage_dir(variant_dir)

    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)

    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)

    if variant in failing_variants:

        if os.path.isfile(spectrum_failed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_failed_coverage_file_dir, VARIANTS_FAILED)

        if os.path.isfile(spectrum_passed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info,search_spaces,
                                                        spectrum_passed_coverage_file_dir, VARIANTS_FAILED)
    else:
        if os.path.isfile(spectrum_failed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_failed_coverage_file_dir, VARIANTS_PASSED)

        if os.path.isfile(spectrum_passed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_passed_coverage_file_dir, VARIANTS_PASSED)

    return features_coverage_info


def read_coverage_info(variant, coverage_info, search_spaces, coverage_file, kind_of_test_count):
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")

        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass')
                    for space in [SS_STMS_IN_F_PRODUCTS, SS_SLICING]:
                        if id not in coverage_info[space]:
                            coverage_info[space][id] = {}
                            coverage_info[space][id][VARIANTS_FAILED] = []
                            coverage_info[space][id][VARIANTS_PASSED] = []
                            coverage_info[space][id][STATEMENT_ID] = []
                        if int(line.get('count')) > 0 and variant not in coverage_info[space][id][kind_of_test_count]:
                            coverage_info[space][id][kind_of_test_count].append(variant)
                        stm_id = line.get('featureClass') + "." + line.get('featureLineNum')
                        if stm_id in search_spaces[space] and stm_id not in coverage_info[space][id][STATEMENT_ID]:
                            coverage_info[space][id][STATEMENT_ID].append(stm_id)
        return coverage_info
    except:
        logging.info("Exception when parsing %s", coverage_file)


def search_rank_worst_case(buggy_stm, features_info, space, spectrum_expression):
    ranking_score_type = spectrum_expression + "_score"
    score = 0
    for feature in features_info:
        if buggy_stm in features_info[feature][STATEMENT_ID]:
            score = features_info[feature][ranking_score_type]
            break

    feature_rank = 0
    stm_rank = 0
    for feature in features_info:
        if features_info[feature][ranking_score_type] >= score:
            feature_rank += 1
            stm_rank += len(space[feature][STATEMENT_ID])
    return feature_rank, stm_rank


def search_rank_best_case(buggy_stm, features_info, spectrum_expression):
    ranking_score_type = spectrum_expression + "_score"
    score = 0
    for feature in features_info:
        if buggy_stm in features_info[feature][STATEMENT_ID]:
            score = features_info[feature][ranking_score_type]
            break

    feature_rank = 0
    stm_rank = 0
    for feature in features_info:
        if features_info[feature][ranking_score_type] > score:
            feature_rank += 1
            stm_rank += len(features_info[feature][STATEMENT_ID])
    return feature_rank + 1, stm_rank + 1


def total_ranking_statements(features_info):
    stms = 0
    for feature in features_info:
        stms += len(features_info[feature][STATEMENT_ID])
    return stms
