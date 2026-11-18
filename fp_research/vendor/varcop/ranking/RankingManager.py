import logging
import os
import time
import xml.etree.ElementTree as ET
from statistics import median, stdev
from scipy import stats
from scipy.stats.mstats import gmean
import numpy

from util.FileManager import join_path, \
    get_test_coverage_dir, get_variant_dir, \
    get_variants_dir, get_all_variant_dirs, list_dir, get_spectrum_failed_coverage_file_name_with_version, \
    get_spectrum_passed_coverage_file_name_with_version

from ranking.Keywords import *
# keywords
from ranking.Spectrum_Expression import *
from util.TestingCoverageManager import statement_coverage
from ranking.VariantLevelRankingManager import calculate_suspiciousness_variant_level, get_num_passing_failing_variants, \
    VARIANT_LEVEL_SUSPICIOUSNESS_SCORE

STM_NOT_FOUND = -1000


def get_set_of_stms(dict_of_stm_per_variant):
    stm_set = []
    for variant in dict_of_stm_per_variant:
        for stm in dict_of_stm_per_variant[variant]:
            if stm not in stm_set:
                stm_set.append(stm)
    return stm_set


def local_ranking_a_suspicious_list(mutated_project_dir, suspicious_stms_list, spectrum_expressions,
                                    spectrum_coverage_prefix):
    local_suspiciousness = {}
    for variant in suspicious_stms_list:
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        statement_info = suspiciousness_calculation(variant_dir, suspicious_stms_list[variant], spectrum_expressions,
                                                    spectrum_coverage_prefix)
        for spectrum_expression in spectrum_expressions:
            if spectrum_expression not in local_suspiciousness.keys():
                local_suspiciousness[spectrum_expression] = {}
            local_suspiciousness[spectrum_expression][variant] = spc_spectrum_ranking(statement_info,
                                                                                      spectrum_expression)
    return local_suspiciousness


def global_ranking_a_suspicious_list(all_stms_of_the_system, all_stms_in_failing_variants, suspicious_stms_list,
                                     local_suspiciousness_of_all_the_system, variant_level_suspiciousness,
                                     spectrum_expression, aggregation_type, normalized_type, alpha=0):
    if normalized_type == NORMALIZATION_ENABLE:
        normalized_score_list = normalize_local_score_alpha_beta(local_suspiciousness_of_all_the_system,
                                                                 all_stms_in_failing_variants,
                                                                 suspicious_stms_list)
    else:
        normalized_score_list = normalize_local_score_none(local_suspiciousness_of_all_the_system,
                                                           all_stms_in_failing_variants, suspicious_stms_list)

    if aggregation_type == AGGREGATION_ARITHMETIC_MEAN:
        ranked_list = global_score_aggregation_arithmetic_mean(all_stms_of_the_system, normalized_score_list,
                                                               variant_level_suspiciousness,
                                                               spectrum_expression, alpha)

    elif aggregation_type == AGGREGATION_GEOMETRIC_MEAN:
        ranked_list = global_score_aggregation_geometric_mean(all_stms_of_the_system, normalized_score_list,
                                                              variant_level_suspiciousness,
                                                              spectrum_expression)
    elif aggregation_type == AGGREGATION_MAX:
        ranked_list = global_score_aggregation_max(all_stms_of_the_system, normalized_score_list,
                                                   variant_level_suspiciousness,
                                                   spectrum_expression)
    elif aggregation_type == AGGREGATION_MIN:
        ranked_list = global_score_aggregation_min(all_stms_of_the_system, normalized_score_list,
                                                   variant_level_suspiciousness,
                                                   spectrum_expression)
    elif aggregation_type == AGGREGATION_MEDIAN:
        ranked_list = global_score_aggregation_median(all_stms_of_the_system, normalized_score_list,
                                                      variant_level_suspiciousness,
                                                      spectrum_expression)

    elif aggregation_type == AGGREGATION_MODE:
        ranked_list = global_score_aggregation_mode(all_stms_of_the_system, normalized_score_list,
                                                    variant_level_suspiciousness,
                                                    spectrum_expression)

    elif aggregation_type == AGGREGATION_STDEV:
        ranked_list = global_score_aggregation_stdev(all_stms_of_the_system, normalized_score_list,
                                                     variant_level_suspiciousness,
                                                     spectrum_expression)

    return ranked_list


def get_all_stms_in_failing_products(all_stms_of_the_system, failing_variants):
    suspicious_stms_list = {}
    for variant in failing_variants:
        suspicious_stms_list[variant] = all_stms_of_the_system[variant]
    return suspicious_stms_list


def product_based_assessment(mutated_project_dir, all_stms_in_failing_products, spectrum_expressions,
                             spectrum_coverage_prefix):
    list_of_stms = get_set_of_stms(all_stms_in_failing_products)
    failing_passing_variants_of_stms, total_fails, total_passes = get_num_passing_failing_variants(mutated_project_dir,
                                                                                                   list_of_stms,
                                                                                                   spectrum_coverage_prefix)
    variant_level_suspiciousness = {}
    for spectrum_expression in spectrum_expressions:
        variant_level_suspiciousness[spectrum_expression] = calculate_suspiciousness_variant_level(
            failing_passing_variants_of_stms,
            total_fails,
            total_passes, spectrum_expression)
        variant_level_suspiciousness[spectrum_expression] = normalized_score(
            variant_level_suspiciousness[spectrum_expression],
            spectrum_expression + VARIANT_LEVEL_SUSPICIOUSNESS_SCORE,
            VARIANT_NUM_OF_FAILS)

    return variant_level_suspiciousness


def sbfl(buggy_statements, mutated_project_dir, search_spaces, spectrum_expressions, spectrum_coverage_prefix,
         coverage_rate):
    stm_info_for_sbfl, total_passed_tests, total_failed_tests = get_infor_for_sbfl(mutated_project_dir,
                                                                                   spectrum_coverage_prefix,
                                                                                   coverage_rate)
    # traditional SBFL
    all_stms_f_products_set = get_set_of_stms(search_spaces[SS_STMS_IN_F_PRODUCTS])
    sliced_stms_set = get_set_of_stms(search_spaces[SS_SLICING])
    full_ranked_list = sbfl_ranking(stm_info_for_sbfl, total_failed_tests, total_passed_tests,
                                    all_stms_f_products_set,
                                    spectrum_expressions)

    isolated_ranked_list = sbfl_ranking(stm_info_for_sbfl, total_failed_tests, total_passed_tests,
                                        sliced_stms_set, spectrum_expressions)
    for metric in spectrum_expressions:
        if metric not in all_buggy_positions.keys():
            all_buggy_positions[metric] = {}

        all_buggy_positions[metric][SBFL_RANK] = locate_multiple_bugs(buggy_statements, sliced_stms_set,
                                                                      full_ranked_list[metric],
                                                                      full_ranked_list[metric])

        all_buggy_positions[metric][SBFL_TC_RANK] = locate_multiple_bugs(buggy_statements, sliced_stms_set,
                                                                        isolated_ranked_list[metric],
                                                                        full_ranked_list[metric])


def varcop(buggy_statements, local_scores, variant_level_suspiciousness, search_spaces,
           spectrum_expressions, aggregation_type, normalized_type, alpha):
    stms_in_f_products_set = get_set_of_stms(search_spaces[SS_STMS_IN_F_PRODUCTS])
    varcop_isolated_set = get_set_of_stms(search_spaces[SS_VARCOP])
    sliced_isolated_set = get_set_of_stms(search_spaces[SS_SLICING])
    for metric in spectrum_expressions:
        if metric not in all_buggy_positions.keys():
            all_buggy_positions[metric] = {}
        full_ranked_list = global_ranking_a_suspicious_list(search_spaces[SS_ALL_STMS],
                                                            search_spaces[SS_STMS_IN_F_PRODUCTS],
                                                            search_spaces[SS_STMS_IN_F_PRODUCTS],
                                                            local_scores[metric],
                                                            variant_level_suspiciousness[metric],
                                                            metric,
                                                            aggregation_type,
                                                            normalized_type, alpha)

        all_buggy_positions[metric][VARCOP_DISABLE_BPC_RANK] = locate_multiple_bugs(buggy_statements,
                                                                                    stms_in_f_products_set,
                                                                                    full_ranked_list,
                                                                                    full_ranked_list)
        # varcop isolation
        varcop_isolated_ranked_list = global_ranking_a_suspicious_list(search_spaces[SS_ALL_STMS],
                                                                       search_spaces[SS_STMS_IN_F_PRODUCTS],
                                                                       search_spaces[SS_VARCOP],
                                                                       local_scores[metric],
                                                                       variant_level_suspiciousness[metric],
                                                                       metric,
                                                                       aggregation_type,
                                                                       normalized_type, alpha)

        all_buggy_positions[metric][VARCOP_RANK] = locate_multiple_bugs(buggy_statements,
                                                                        varcop_isolated_set,
                                                                        varcop_isolated_ranked_list,
                                                                        full_ranked_list)

        # tc_based_isolation
        sliced_isolated_ranked_list = global_ranking_a_suspicious_list(search_spaces[SS_ALL_STMS],
                                                                       search_spaces[SS_STMS_IN_F_PRODUCTS],
                                                                       search_spaces[SS_SLICING],
                                                                       local_scores[metric],
                                                                       variant_level_suspiciousness[metric],
                                                                       metric,
                                                                       aggregation_type,
                                                                       normalized_type, alpha)

        all_buggy_positions[metric][VARCOP_TC_RANK] = locate_multiple_bugs(buggy_statements,
                                                                          sliced_isolated_set,
                                                                          sliced_isolated_ranked_list,
                                                                          full_ranked_list)


def ranking_multiple_bugs(buggy_statements, mutated_project_dir, search_spaces, spectrum_expressions,
                          aggregation_type, normalized_type, spectrum_coverage_prefix="", coverage_rate=0.0, alpha=0):
    start_time = time.time()

    global all_buggy_positions
    all_buggy_positions = {}

    variant_level_suspiciousness = product_based_assessment(mutated_project_dir, search_spaces[SS_ALL_STMS],
                                                            spectrum_expressions, spectrum_coverage_prefix)

    local_suspiciousness_of_all_the_system = local_ranking_a_suspicious_list(mutated_project_dir,
                                                                             search_spaces[SS_STMS_IN_F_PRODUCTS],
                                                                             spectrum_expressions,
                                                                             spectrum_coverage_prefix)
    sbfl(buggy_statements, mutated_project_dir, search_spaces, spectrum_expressions, spectrum_coverage_prefix,
         coverage_rate)
    varcop(buggy_statements, local_suspiciousness_of_all_the_system, variant_level_suspiciousness, search_spaces,
           spectrum_expressions, aggregation_type, normalized_type, alpha)

    varcop_ranking_time = time.time() - start_time
    return all_buggy_positions, varcop_ranking_time


def locate_multiple_bugs(buggy_statements, isolated_set, isolated_ranked_list,
                         full_ranked_list):
    bugs = {}
    for stm in buggy_statements:
        bugs[stm] = {}
        bugs[stm][RANK] = search_rank_worst_case_by_layer(stm, isolated_ranked_list)
        if bugs[stm][RANK] == STM_NOT_FOUND:
            without_isolation_buggy_position = search_rank_worst_case_by_layer(stm, full_ranked_list)
            bugs[stm][RANK] = without_isolation_buggy_position + len(isolated_set)
            for i in range(0, without_isolation_buggy_position):
                if full_ranked_list[i][0] in isolated_set:
                    bugs[stm][RANK] -= 1
            bugs[stm][EXAM] = (bugs[stm][RANK] / len(full_ranked_list)) * 100
        else:
            bugs[stm][EXAM] = bugs[stm][RANK] / len(isolated_ranked_list)
    return bugs


def get_local_score(stm, ranked_list):
    for i in range(0, len(ranked_list)):
        if stm == ranked_list[i][0]:
            return ranked_list[i][1], ranked_list[i][2]
    return STM_NOT_FOUND, 0


def normalized_score(scores_list, normalized_value, failings, alpha=0, beta=1):
    min_score = 1000000
    max_score = -1000000
    for stm in scores_list:
        if scores_list[stm][normalized_value] > max_score and scores_list[stm][failings] != 0:
            max_score = scores_list[stm][normalized_value]

        if (scores_list[stm][normalized_value] < min_score):
            min_score = scores_list[stm][normalized_value]
    for stm in scores_list:
        tmp = scores_list[stm][normalized_value]
        if min_score == max_score:
            scores_list[stm][normalized_value] = alpha
        else:
            if scores_list[stm][failings] == 0:
                scores_list[stm][normalized_value] = alpha
            else:
                scores_list[stm][normalized_value] = (tmp - min_score) * (
                        (beta - alpha) / (max_score - min_score)) + alpha
    return scores_list


def normalize_local_score_alpha_beta(local_suspiciousness_of_all_the_system, all_statements_in_failing_variants,
                                     suspicious_stms_list, alpha=0, beta=1):
    all_suspicious_stm = get_set_of_stms(suspicious_stms_list)

    normalized_score_list = {}
    for variant in all_statements_in_failing_variants:
        normalized_score_list[variant] = {}
        if (len(all_statements_in_failing_variants[variant]) > 0):
            max = local_suspiciousness_of_all_the_system[variant][0][1]
            min = \
                local_suspiciousness_of_all_the_system[variant][
                    len(local_suspiciousness_of_all_the_system[variant]) - 1][1]
            for stm in all_suspicious_stm:
                local_score, num_of_failing_test = get_local_score(stm, local_suspiciousness_of_all_the_system[variant])

                if local_score == STM_NOT_FOUND:
                    normalized_score = alpha
                else:
                    if min == max:
                        normalized_score = beta
                    else:
                        if num_of_failing_test == 0:
                            normalized_score = alpha
                        else:
                            normalized_score = (local_score - min) * ((beta - alpha) / (max - min)) + alpha
                normalized_score_list[variant][stm] = normalized_score
    return normalized_score_list


def normalize_local_score_none(local_suspiciousness_of_all_the_system, all_statements_in_failing_variants,
                               suspicious_stms_list):
    all_suspicious_stm = get_set_of_stms(suspicious_stms_list)
    normalized_score_list = {}
    for variant in all_statements_in_failing_variants:
        normalized_score_list[variant] = {}
        if len(all_statements_in_failing_variants[variant]) > 0:
            for stm in all_suspicious_stm:
                local_score, num_of_fail_test = get_local_score(stm, local_suspiciousness_of_all_the_system[variant])
                if local_score == STM_NOT_FOUND:
                    normalized_score = 0
                else:
                    normalized_score = local_score
                normalized_score_list[variant][stm] = normalized_score
    return normalized_score_list


def count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system, normalized_score_list):
    for stm in all_stms_score_list:
        all_stms_score_list[stm][NUM_OF_PASSING_VARIANTS] = 0
        for variant in all_stms_of_the_system:
            if variant not in normalized_score_list.keys() and stm in all_stms_of_the_system[variant]:
                all_stms_score_list[stm][NUM_OF_PASSING_VARIANTS] += 1
    return all_stms_score_list


def global_score_aggregation_min(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                 spectrum_expression):
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in all_stms_score_list:
                if (all_stms_score_list[stm][score_type] > normalized_score_list[variant][stm]):
                    all_stms_score_list[stm][score_type] = normalized_score_list[variant][stm]
            if stm not in all_stms_score_list:
                all_stms_score_list[stm] = {}
                all_stms_score_list[stm][score_type] = normalized_score_list[variant][stm]

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)

    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def global_score_aggregation_max(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                 spectrum_expression):
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in all_stms_score_list:
                if (all_stms_score_list[stm][score_type] < normalized_score_list[variant][stm]):
                    all_stms_score_list[stm][score_type] = normalized_score_list[variant][stm]
            if stm not in all_stms_score_list:
                all_stms_score_list[stm] = {}
                all_stms_score_list[stm][score_type] = normalized_score_list[variant][stm]

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def global_score_aggregation_median(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                    spectrum_expression):
    list_of_scores = {}
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in list_of_scores:
                list_of_scores[stm].append(normalized_score_list[variant][stm])
            if stm not in list_of_scores:
                list_of_scores[stm] = []
                list_of_scores[stm].append(normalized_score_list[variant][stm])

    for stm in list_of_scores.keys():
        all_stms_score_list[stm] = {}
        all_stms_score_list[stm][score_type] = median(list_of_scores[stm])

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def global_score_aggregation_stdev(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                   spectrum_expression):
    list_of_scores = {}
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in list_of_scores:
                list_of_scores[stm].append(normalized_score_list[variant][stm])
            if stm not in list_of_scores:
                list_of_scores[stm] = []
                list_of_scores[stm].append(normalized_score_list[variant][stm])

    for stm in list_of_scores.keys():
        all_stms_score_list[stm] = {}
        if len(list_of_scores[stm]) < 2:
            all_stms_score_list[stm][score_type] = list_of_scores[stm][0]
        else:
            all_stms_score_list[stm][score_type] = stdev(list_of_scores[stm])

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def global_score_aggregation_mode(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                  spectrum_expression):
    list_of_scores = {}
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in list_of_scores:
                list_of_scores[stm].append(normalized_score_list[variant][stm])
            if stm not in list_of_scores:
                list_of_scores[stm] = []
                list_of_scores[stm].append(normalized_score_list[variant][stm])

    for stm in list_of_scores.keys():
        all_stms_score_list[stm] = {}
        all_stms_score_list[stm][score_type] = stats.mode(numpy.array(list_of_scores[stm])).mode[0]

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def global_score_aggregation_arithmetic_mean(all_stms_of_the_system, normalized_score_list,
                                             variant_level_suspiciousness, spectrum_expression, alpha=0):
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in all_stms_score_list:
                all_stms_score_list[stm][score_type] += normalized_score_list[variant][stm]
                all_stms_score_list[stm][NUM_OF_FAILING_VARIANTS] += 1
            if stm not in all_stms_score_list:
                all_stms_score_list[stm] = {}
                all_stms_score_list[stm][score_type] = normalized_score_list[variant][stm]
                all_stms_score_list[stm][NUM_OF_FAILING_VARIANTS] = 1

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)

    for stm in all_stms_score_list:
        all_stms_score_list[stm][score_type] = all_stms_score_list[stm][score_type] / all_stms_score_list[stm][
            NUM_OF_FAILING_VARIANTS]

    all_stms_score_list = normalized_score(all_stms_score_list, score_type, NUM_OF_FAILING_VARIANTS)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression, alpha)


def global_score_aggregation_geometric_mean(all_stms_of_the_system, normalized_score_list, variant_level_suspiciousness,
                                            spectrum_expression):
    list_of_scores = {}
    score_type = spectrum_expression + "_score"
    all_stms_score_list = {}
    for variant in normalized_score_list:
        for stm in normalized_score_list[variant]:
            if stm in list_of_scores:
                list_of_scores[stm].append(normalized_score_list[variant][stm])
            if stm not in list_of_scores:
                list_of_scores[stm] = []
                list_of_scores[stm].append(normalized_score_list[variant][stm])

    for stm in list_of_scores.keys():
        all_stms_score_list[stm] = {}
        all_stms_score_list[stm][score_type] = gmean(list_of_scores[stm])

    all_stms_score_list = count_num_of_passing_products_for_a_stm(all_stms_score_list, all_stms_of_the_system,
                                                                  normalized_score_list)
    return varcop_ranking(all_stms_score_list, variant_level_suspiciousness, spectrum_expression)


def sbfl_ranking(stm_info_for_sbfl, total_failed_tests, total_passed_tests, isolated_stms_set, spectrum_expressions):
    spectrum_ranked_list = {}
    for spectrum_expression in spectrum_expressions:
        statement_with_scores = spectrum_calculation(stm_info_for_sbfl, total_failed_tests,
                                                     total_passed_tests,
                                                     spectrum_expression)

        spectrum_ranked_list[spectrum_expression] = sbfl_rank_by_metric(statement_with_scores, isolated_stms_set,
                                                                        spectrum_expression)

    return spectrum_ranked_list


def get_executed_stms_of_the_system(mutated_project_dir, spectrum_coverage_prefix, coverage_rate):
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_list = list_dir(variants_dir)
    all_stms_in_system = {}
    all_stms_in_failing_product = {}
    for variant in variants_list:
        stm_coverage = 0
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        if coverage_rate > 0:
            stm_coverage = statement_coverage(variant_dir, spectrum_coverage_prefix)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        failed_file = join_path(test_coverage_dir, spectrum_failed_file)
        spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
        passed_file = join_path(test_coverage_dir, spectrum_passed_file)

        # if variant is a passing coverage and statement coverage is less than the coverage rate
        if not os.path.isfile(failed_file) and coverage_rate != 0 and stm_coverage <= coverage_rate:
            continue

        coverage_files = [failed_file, passed_file]

        for c_file in coverage_files:
            if os.path.isfile(c_file):
                data = {}
                try:
                    tree = ET.parse(c_file)
                    root = tree.getroot()
                    project = root.find("project")

                    for package in project:
                        for file in package:
                            for line in file:
                                id = line.get('featureClass') + "." + line.get('featureLineNum')
                                if id not in data:
                                    data[id] = {'num_interactions': 0}
                except:
                    logging.info("Exception when parsing %s", c_file)
                all_stms_in_system[variant] = data
                if coverage_files.index(c_file) == 0:
                    all_stms_in_failing_product[variant] = data
    return all_stms_in_system, all_stms_in_failing_product


def get_infor_for_sbfl(mutated_project_dir, spectrum_coverage_prefix, coverage_rate):
    total_failed_tests = 0
    total_passed_tests = 0
    stm_info_for_spectrum = {}
    variants_list = get_all_variant_dirs(mutated_project_dir)
    for variant_dir in variants_list:
        stm_coverage = 0
        if coverage_rate > 0:
            stm_coverage = statement_coverage(variant_dir, spectrum_coverage_prefix)
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)
        spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)

        # if variant is a passing variant and stm_coverage < coverage_rate
        if not os.path.isfile(
                spectrum_failed_coverage_file_dir) and coverage_rate != 0 and stm_coverage <= coverage_rate:
            continue
        if os.path.isfile(spectrum_failed_coverage_file_dir):
            stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                    spectrum_failed_coverage_file_dir,
                                                                    FAILED_TEST_COUNT)

        if os.path.isfile(spectrum_passed_coverage_file_dir):
            stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                    spectrum_passed_coverage_file_dir,
                                                                    PASSED_TEST_COUNT)

        ftests, ptests = count_tests(test_coverage_dir, spectrum_coverage_prefix)
        total_failed_tests += ftests
        total_passed_tests += ptests

    return stm_info_for_spectrum, total_passed_tests, total_failed_tests


def read_coverage_info_for_spectrum(statement_infor, coverage_file, kind_of_test_count):
    data = {}
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")

        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass') + "." + line.get('featureLineNum')
                    if id not in data:
                        data[id] = {}
                        data[id][FAILED_TEST_COUNT] = 0
                        data[id][PASSED_TEST_COUNT] = 0
                    data[id][kind_of_test_count] = max(int(line.get('count')),
                                                       data[id][kind_of_test_count])

        for id in data.keys():
            if id not in statement_infor:
                statement_infor[id] = data[id]
            else:
                statement_infor[id][kind_of_test_count] += data[id][kind_of_test_count]

        return statement_infor
    except:
        logging.info("Exception when parsing %s", coverage_file)


def suspiciousness_calculation(variant_dir, suspicious_stms_list, spectrum_expressions, spectrum_coverage_prefix):
    statement_infor = {}
    test_coverage_dir = get_test_coverage_dir(variant_dir)

    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)
    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)

    if os.path.isfile(spectrum_failed_coverage_file_dir):
        statement_infor = read_statement_infor_from_coverage_file(statement_infor, spectrum_failed_coverage_file_dir,
                                                                  FAILED_TEST_COUNT, suspicious_stms_list)

    if os.path.isfile(spectrum_passed_coverage_file_dir):
        statement_infor = read_statement_infor_from_coverage_file(statement_infor, spectrum_passed_coverage_file_dir,
                                                                  PASSED_TEST_COUNT, suspicious_stms_list)

    (total_failed_tests, total_passed_tests) = count_tests(test_coverage_dir, spectrum_coverage_prefix)
    for spectrum_expression in spectrum_expressions:
        statement_infor = spectrum_calculation(statement_infor, total_failed_tests, total_passed_tests,
                                               spectrum_expression)

    return statement_infor


def count_test_in_file(file_dir):
    try:
        tree = ET.parse(file_dir)
        root = tree.getroot()
        project = root.find("tests")
        return int(project.get("count"))
    except:
        logging.info("Exception when parsing %s", file_dir)


def count_tests(dir, spectrum_coverage_prefix):
    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(dir, spectrum_failed_file)
    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(dir, spectrum_passed_file)

    num_of_failed_tests = 0
    num_of_passed_tests = 0
    if os.path.isfile(spectrum_failed_coverage_file_dir):
        num_of_failed_tests = count_test_in_file(spectrum_failed_coverage_file_dir)

    if os.path.isfile(spectrum_passed_coverage_file_dir):
        num_of_passed_tests = count_test_in_file(spectrum_passed_coverage_file_dir)

    return num_of_failed_tests, num_of_passed_tests


def read_statement_infor_from_coverage_file(statement_infor, coverage_file, kind_of_test_count, suspicious_stms_list):
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")

        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass') + "." + line.get('featureLineNum')
                    if id not in statement_infor:
                        statement_infor[id] = {}
                        statement_infor[id][FAILED_TEST_COUNT] = 0
                        statement_infor[id][PASSED_TEST_COUNT] = 0

                        if id in suspicious_stms_list.keys():
                            statement_infor[id][SUSPICIOUS] = True
                        else:
                            statement_infor[id][SUSPICIOUS] = False
                    statement_infor[id][kind_of_test_count] = max(int(line.get('count')),
                                                                  statement_infor[id][kind_of_test_count])
        return statement_infor
    except:
        logging.info("Exception when parsing %s", coverage_file)


def spectrum_calculation(statement_infor, total_failed_tests, total_passed_tests, spectrum_expression):
    for id in statement_infor.keys():
        if spectrum_expression == TARANTULA:

            statement_infor[id][TARANTULA_SCORE] = tarantula_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                         statement_infor[id][PASSED_TEST_COUNT],
                                                                         total_failed_tests,
                                                                         total_passed_tests)

        elif spectrum_expression == OCHIAI:
            statement_infor[id][OCHIAI_SCORE] = ochiai_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == OP2:
            statement_infor[id][OP2_SCORE] = op2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                             statement_infor[id][PASSED_TEST_COUNT], total_failed_tests,
                                                             total_passed_tests)

        elif spectrum_expression == BARINEL:
            statement_infor[id][BARINEL_SCORE] = barinel_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                     statement_infor[id][PASSED_TEST_COUNT],
                                                                     total_failed_tests,
                                                                     total_passed_tests)

        elif spectrum_expression == DSTAR:
            statement_infor[id][DSTAR_SCORE] = dstar_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == RUSSELL_RAO:
            statement_infor[id][RUSSELL_RAO_SCORE] = russell_rao_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                             statement_infor[id][PASSED_TEST_COUNT],
                                                                             total_failed_tests, total_passed_tests)
        elif spectrum_expression == SIMPLE_MATCHING:
            statement_infor[id][SIMPLE_MATCHING_SCORE] = simple_matching_calculation(
                statement_infor[id][FAILED_TEST_COUNT], statement_infor[id][PASSED_TEST_COUNT],
                total_failed_tests, total_passed_tests)
        elif spectrum_expression == ROGERS_TANIMOTO:
            statement_infor[id][ROGERS_TANIMOTO_SCORE] = rogers_tanimoto_calculation(
                statement_infor[id][FAILED_TEST_COUNT], statement_infor[id][PASSED_TEST_COUNT],
                total_failed_tests, total_passed_tests)
        elif spectrum_expression == AMPLE:
            statement_infor[id][AMPLE_SCORE] = ample2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                  statement_infor[id][PASSED_TEST_COUNT],
                                                                  total_failed_tests, total_passed_tests)

        elif spectrum_expression == JACCARD:
            statement_infor[id][JACCARD_SCORE] = jaccard_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                     statement_infor[id][PASSED_TEST_COUNT],
                                                                     total_failed_tests, total_passed_tests)
        elif spectrum_expression == COHEN:
            statement_infor[id][COHEN_SCORE] = cohen_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)

        elif spectrum_expression == SCOTT:
            statement_infor[id][SCOTT_SCORE] = scott_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == ROGOT1:
            statement_infor[id][ROGOT1_SCORE] = rogot1_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)

        elif spectrum_expression == GEOMETRIC_MEAN:

            statement_infor[id][GEOMETRIC_MEAN_SCORE] = geometric_mean_calculation(
                statement_infor[id][FAILED_TEST_COUNT],
                statement_infor[id][PASSED_TEST_COUNT],
                total_failed_tests, total_passed_tests)

        elif spectrum_expression == M2:
            statement_infor[id][M2_SCORE] = m2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                           statement_infor[id][PASSED_TEST_COUNT],
                                                           total_failed_tests, total_passed_tests)
        elif spectrum_expression == WONG1:
            statement_infor[id][WONG1_SCORE] = wong1_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)

        elif spectrum_expression == SOKAL:
            statement_infor[id][SOKAL_SCORE] = sokal_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)
        # new
        elif spectrum_expression == SORENSEN_DICE:

            statement_infor[id][SORENSEN_DICE_SCORE] = sorensen_dice_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == DICE:

            statement_infor[id][DICE_SCORE] = dice_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                               statement_infor[id][PASSED_TEST_COUNT],
                                                               total_failed_tests, total_passed_tests)

        elif spectrum_expression == HUMANN:

            statement_infor[id][HUMANN_SCORE] = humman_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == M1:

            statement_infor[id][M1_SCORE] = m1_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                           statement_infor[id][PASSED_TEST_COUNT],
                                                           total_failed_tests, total_passed_tests)
        elif spectrum_expression == WONG2:

            statement_infor[id][WONG2_SCORE] = wong2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == WONG3:

            statement_infor[id][WONG3_SCORE] = wong3_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == ZOLTAR:

            statement_infor[id][ZOLTAR_SCORE] = zoltar_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == OVERLAP:

            statement_infor[id][OVERLAP_SCORE] = overlap_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                     statement_infor[id][PASSED_TEST_COUNT],
                                                                     total_failed_tests, total_passed_tests)
        elif spectrum_expression == EUCLID:

            statement_infor[id][EUCLID_SCORE] = euclid_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == ROGOT2:

            statement_infor[id][ROGOT2_SCORE] = rogot2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == HAMMING:

            statement_infor[id][HAMMING_SCORE] = hamming_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                     statement_infor[id][PASSED_TEST_COUNT],
                                                                     total_failed_tests, total_passed_tests)
        elif spectrum_expression == FLEISS:

            statement_infor[id][FLEISS_SCORE] = fleiss_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                   statement_infor[id][PASSED_TEST_COUNT],
                                                                   total_failed_tests, total_passed_tests)
        elif spectrum_expression == ANDERBERG:
            statement_infor[id][ANDERBERG_SCORE] = anderberg_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                         statement_infor[id][PASSED_TEST_COUNT],
                                                                         total_failed_tests, total_passed_tests)

        elif spectrum_expression == GOODMAN:
            statement_infor[id][GOODMAN_SCORE] = goodman_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                     statement_infor[id][PASSED_TEST_COUNT],
                                                                     total_failed_tests, total_passed_tests)

        elif spectrum_expression == HARMONIC_MEAN:

            statement_infor[id][HARMONIC_MEAN_SCORE] = harmonic_mean_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                                 statement_infor[id][PASSED_TEST_COUNT],
                                                                                 total_failed_tests, total_passed_tests)
        elif spectrum_expression == KULCZYNSKI1:

            statement_infor[id][KULCZYNSKI1_SCORE] = kulczynski1_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                             statement_infor[id][PASSED_TEST_COUNT],
                                                                             total_failed_tests, total_passed_tests)
        elif spectrum_expression == KULCZYNSKI2:

            statement_infor[id][KULCZYNSKI2_SCORE] = kulczynski2_calculation(statement_infor[id][FAILED_TEST_COUNT],
                                                                             statement_infor[id][PASSED_TEST_COUNT],
                                                                             total_failed_tests, total_passed_tests)

    return statement_infor


def sbfl_rank_by_metric(statements_infor, isolated_stms, spectrum_expression):
    spectrum_ranked_list = []

    score_type = spectrum_expression + "_score"
    for (key, value) in statements_infor.items():
        if key in isolated_stms:
            spectrum_ranked_list.append(
                (key, statements_infor[key][score_type], statements_infor[key][FAILED_TEST_COUNT]))

    # 0 is the position of key, 1 is the position of score
    return descending_sort(data=spectrum_ranked_list, sorted_element=1)


def varcop_ranking(statements_infor, variant_level_suspiciousness, spectrum_expression, alpha=0.5):
    ranked_list = []

    score_type = spectrum_expression + "_score"
    for (key, value) in statements_infor.items():
        score_tmp = alpha * statements_infor[key][score_type] + (1 - alpha) * variant_level_suspiciousness[key][
            spectrum_expression + VARIANT_LEVEL_SUSPICIOUSNESS_SCORE]
        ranked_list.append((key, score_tmp, statements_infor[key][NUM_OF_PASSING_VARIANTS]))
    return descending_sort(data=ranked_list, sorted_element=1)


def spc_spectrum_ranking(statements_infor, spectrum_expression):
    spc_spectrum_ranked_list = []
    score_type = spectrum_expression + "_score"

    for (key, value) in statements_infor.items():
        # if statements_infor[key][SUSPICIOUS]:
        spc_spectrum_ranked_list.append(
            (key, statements_infor[key][score_type], statements_infor[key][FAILED_TEST_COUNT]))

    # 0 is the position of key, 1 is the position of score
    return descending_sort(data=spc_spectrum_ranked_list, sorted_element=1)


def descending_sort(data, sorted_element):
    for i in range(0, len(data) - 1):
        for j in range(i + 1, len(data)):
            if data[i][sorted_element] < data[j][sorted_element]:
                data[i], data[j] = \
                    data[j], data[i]

    return data


def ascending_sort(data, sorted_element):
    for i in range(0, len(data) - 1):
        for j in range(i + 1, len(data)):
            if data[i][sorted_element] > data[j][sorted_element]:
                data[i], data[j] = \
                    data[j], data[i]

    return data


def search_rank_worst_case(stm, ranked_list):
    for i in range(0, len(ranked_list)):
        if ranked_list[i][0] == stm:
            j = i
            while j < len(ranked_list) - 1:
                if ranked_list[j][1] == ranked_list[j + 1][1]:
                    j += 1
                else:
                    break
            return j + 1
    return STM_NOT_FOUND


def search_rank_worst_case_by_layer(stm, ranked_list):
    for i in range(0, len(ranked_list)):
        if ranked_list[i][0] == stm:
            j = i
            while j < len(ranked_list) - 1:
                if ranked_list[j][1] == ranked_list[j + 1][1] and ranked_list[j][2] == ranked_list[j + 1][2]:
                    j += 1
                else:
                    break
            return j + 1
    return STM_NOT_FOUND
