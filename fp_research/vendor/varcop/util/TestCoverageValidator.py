import xml.etree.ElementTree as ET
from collections import defaultdict

from util import TestManager
from util.FileManager import get_all_variant_dirs, get_test_dir, \
    get_all_coverage_file_paths_in_dir, get_test_coverage_dir, get_file_name, TEST_COVERAGE_FILE_EXTENSION, \
    get_failed_test_coverage_dir, get_failed_spectrum_coverage_file_path_with_version, \
    get_passed_spectrum_coverage_file_path_with_version, get_passed_test_coverage_dir
from Helpers import execute_shell_command, get_logger
from util.TestManager import check_variant_final_test_output
from suspicious_statements_manager import SuspiciousStatementManager

logger = get_logger(__name__)


def check(mutated_project_dir):
    logger.info(f"Checking test case & coverage consistency - {mutated_project_dir}")
    mutated_project_name = get_file_name(mutated_project_dir)
    feature_buggy_statements = SuspiciousStatementManager.get_multiple_buggy_statements(mutated_project_name,
                                                                                        mutated_project_dir)
    variant_dirs = get_all_variant_dirs(mutated_project_dir, sort=True)
    for variant_dir in variant_dirs:
        check_test_result_consistency_between_batch_test_and_single_test(variant_dir)
        check_coverage_file_quantity(variant_dir)
        check_buggy_statements_executed_in_failed_tests(variant_dir, feature_buggy_statements)
        check_spectrum_coverage_file_aggregated_from_all_tests(variant_dir)


def check_test_result_consistency_between_batch_test_and_single_test(variant_dir):
    final_test_output_flag = check_variant_final_test_output(variant_dir,
                                                             junit_mode=TestManager.JunitMode.FULL_COVERAGE)
    assert final_test_output_flag is not None, f"\n---\nInconsistent in test result of batch test and single test [JUNIT]\nVariant {variant_dir}\n---"


def check_spectrum_coverage_file_aggregated_from_all_tests(variant_dir):
    test_coverage_dir = get_test_coverage_dir(variant_dir)

    passed_coverage_dir = get_passed_test_coverage_dir(variant_dir)
    if passed_coverage_dir is None:
        return
    passed_spectrum_coverage_file = get_passed_spectrum_coverage_file_path_with_version(test_coverage_dir)
    validate_spectrum_coverage_aggregated_completely(passed_spectrum_coverage_file, passed_coverage_dir)

    failed_coverage_dir = get_failed_test_coverage_dir(variant_dir)
    if failed_coverage_dir is None:
        return
    failed_spectrum_coverage_file = get_failed_spectrum_coverage_file_path_with_version(test_coverage_dir)
    validate_spectrum_coverage_aggregated_completely(failed_spectrum_coverage_file, failed_coverage_dir)


def validate_spectrum_coverage_aggregated_completely(spectrum_coverage_file, by_result_coverage_dir):
    tree = ET.parse(spectrum_coverage_file)
    root = tree.getroot()
    aggregated_coverage_file_names = set([e.get("source") for e in root.findall("tests/test")])
    coverage_file_names = set([get_file_name(e) for e in get_all_coverage_file_paths_in_dir(by_result_coverage_dir)])
    assert aggregated_coverage_file_names == coverage_file_names, f"\n---\nAggregated coverage file names: {aggregated_coverage_file_names}\nCoverage file names: {coverage_file_names}\n\nCoverage dir: {by_result_coverage_dir}\n---"


def check_buggy_statements_executed_in_failed_tests(variant_dir, feature_buggy_statements):
    test_coverage_dir = get_test_coverage_dir(variant_dir)
    failed_coverage_dir = get_failed_test_coverage_dir(variant_dir)
    if failed_coverage_dir is None:
        return
    buggy_statement_dict = get_buggy_statements_in_variant_source_code(test_coverage_dir, feature_buggy_statements)
    for failed_coverage_file_path in get_all_coverage_file_paths_in_dir(failed_coverage_dir):
        validate_buggy_statement_count_in_coverage_file(failed_coverage_file_path, buggy_statement_dict)


def validate_buggy_statement_count_in_coverage_file(coverage_file_path, buggy_statement_dict):
    tree = ET.parse(coverage_file_path)
    root = tree.getroot()
    project = root.find("project")
    buggy_statement_executed_flag = False
    for package in project:
        if package.tag != "package":
            continue
        for file in package:
            if file.tag != "file":
                continue
            full_file_path = file.get("path")
            for relative_file_path in buggy_statement_dict:
                if full_file_path.endswith("src/" + relative_file_path):
                    line_nums = buggy_statement_dict[relative_file_path]
                    for line in file:
                        if line.tag != "line":
                            continue
                        if line.get("type") not in ["stmt", "method"]:
                            continue
                        if line.get("num") in line_nums and int(line.get("count")) >= 0:
                            buggy_statement_executed_flag = True
                    break

    assert buggy_statement_executed_flag, f"\n---\nReal buggy statement: {buggy_statement_dict}\nCoverage file path: {coverage_file_path}\n---"


def get_buggy_statements_in_variant_source_code(test_coverage_dir, buggy_statements):
    failed_spectrum_coverage_file = get_failed_spectrum_coverage_file_path_with_version(test_coverage_dir)
    buggy_statement_dict = {}
    related_source_file_names = set()

    for s in buggy_statements:
        split_s = s.rsplit(".", 1)
        related_feature_file_name = split_s[0]
        related_feature_line_num = split_s[1]
        buggy_statement_dict[related_feature_file_name] = related_feature_line_num
        related_source_file_names.add(related_feature_file_name.split(".", 1)[1])

    real_buggy_statement_dict = defaultdict(set)
    tree = ET.parse(failed_spectrum_coverage_file)
    root = tree.getroot()
    project = root.find("project")
    for package in project:
        if package.tag != "package":
            continue
        for file in package:
            if file.tag != "file":
                continue
            if file.get("name") not in related_source_file_names:
                continue
            for line in file:
                if line.tag != "line":
                    continue
                if line.get("type") not in ["stmt", "method"]:
                    continue
                if buggy_statement_dict.get(line.get('featureClass'), "_") == line.get('featureLineNum'):
                    real_line_num = line.get("num")
                    real_source_file_path = file.get("path")
                    real_buggy_statement_dict[real_source_file_path].add(real_line_num)
    return dict(real_buggy_statement_dict)


def check_coverage_file_quantity(variant_dir):
    test_case_names = get_all_test_case_name_from_source_files(variant_dir)
    coverage_dir = get_test_coverage_dir(variant_dir)
    coverage_file_names = set([get_file_name(e).replace(f".{TEST_COVERAGE_FILE_EXTENSION}", "") for e in
                               get_all_coverage_file_paths_in_dir(coverage_dir)])
    assert test_case_names == coverage_file_names, f"\n---\nTest case names: {test_case_names}\nCoverage file names: {coverage_file_names}\n\nVariant dir: {variant_dir}\n---"


def get_all_test_case_name_from_source_files(variant_dir):
    test_dir = get_test_dir(variant_dir) + "/"
    shell_command = f"""find {test_dir} -name "*_ESTest.java" -exec egrep -oH "void test[0-9]+\(\)  throws" {{}} \; | sed 's|{test_dir}||1' | sed 's/^\///1; s/\//./g; s/\.java:void /./1; s/()  throws//1;'"""
    output = execute_shell_command(shell_command, show_command=False)
    test_cases_names = output.split("\n")[0:-1]
    test_cases_name_set = set(test_cases_names)
    assert len(test_cases_names) == len(
        test_cases_name_set), f"\n---\nDuplicate test case names\nBefore [LIST]: {test_cases_names}\nAfter [SET]: {test_cases_name_set}\n\nVariant dir: {variant_dir}\n---"
    return test_cases_name_set
