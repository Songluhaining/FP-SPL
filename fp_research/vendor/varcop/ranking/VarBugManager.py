import csv
import os

from util.FileManager import get_variants_dir, list_dir, get_variant_dir, get_test_coverage_dir, join_path, \
    get_spectrum_failed_coverage_file_name_with_version, get_model_configs_report_path
from util.TestingCoverageManager import statement_coverage


def is_var_bug(mutated_project_dir, filter_coverage, spectrum_coverage_prefix=""):
    num_of_failing_variants = 0
    num_of_passing_variants = 0
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_list = list_dir(variants_dir)
    for variant in variants_list:
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        stm_coverage = statement_coverage(variant_dir, spectrum_coverage_prefix)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        failed_file = join_path(test_coverage_dir, spectrum_failed_file)
        if os.path.isfile(failed_file):
            num_of_failing_variants += 1
        elif stm_coverage >= filter_coverage:
            num_of_passing_variants += 1

    if num_of_failing_variants >= 1 and num_of_passing_variants >= 1:
        return 1
    return 0


def is_var_bug_by_config(mutated_project_dir, bases):
    config_report_path = get_model_configs_report_path(mutated_project_dir)

    with open(config_report_path) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        feature_names = header[1:]
        for row in reader:
            if row[-1] == "__FAILED__":
                flag = True
                for i in range(0, len(feature_names)):

                    if row[i + 1].strip() == "T":
                        if feature_names[i] not in bases:
                            flag = False

                if flag:
                    return 0
    return 1
