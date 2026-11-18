import csv
import json

from util import AntManager
import ConfigManager
import TestManager
import VariantComposer
from FileManager import get_all_variant_dirs, get_model_configs_report_path, get_file_name, get_file_name_without_ext, \
    get_outer_dir, join_path, get_dependency_lib_dirs, is_path_exist, touch_file, \
    get_variant_dir_from_config_path, get_feature_order_file_path
from suspicious_statements_manager.SuspiciousStatementManager import get_mutated_features

SWITCHED_VARIANT_NAME_POSTFIX = "SW"
SW_PASSED_TEST_FLAG_FILE_NAME = "sw.test.passed.txt"
SW_FAILED_TEST_FLAG_FILE_NAME = "sw.test.failed.txt"


def find_involving_features(project_dir, mutated_project_dir, custom_ant):
    failed_variant_dir_items = get_failed_variant_dir_items(mutated_project_dir)
    for item in failed_variant_dir_items:
        failed_variant_name, failed_variant_dir = item
        related_config_path = get_related_config_path(failed_variant_name, project_dir)
        switched_config_paths = compose_switched_configs(related_config_path, failed_variant_name)
        compose_switched_products(switched_config_paths, project_dir, mutated_project_dir, custom_ant=custom_ant)


def get_failed_variant_dir_items(mutated_project_dir):
    variant_dirs = get_all_variant_dirs(mutated_project_dir, sort=True)
    configs_report_file_path = get_model_configs_report_path(mutated_project_dir)
    failed_variant_dir_items = []
    with open(configs_report_file_path, "r") as report_csv:
        reader = csv.reader(report_csv)
        next(reader)
        for i, row in enumerate(reader):
            test_passed = row[-1] == "__PASSED__"
            if not test_passed:
                failed_variant_name = row[0]
                for d in variant_dirs:
                    split_variant_name = get_file_name(d)
                    if split_variant_name == failed_variant_name:
                        failed_variant_dir_items.append((failed_variant_name, d))
    return failed_variant_dir_items


def get_related_config_path(variant_name, project_dir):
    config_paths = ConfigManager.get_config_paths(project_dir, sort=True)
    for p in config_paths:
        split_config_name = get_file_name_without_ext(p)
        if split_config_name == variant_name:
            return p


def compose_switched_configs(config_path, variant_name):
    feature_selections = get_feature_selections(config_path)
    switched_feature_selections_container = get_single_switched_feature_selections(feature_selections)
    configs_dir = get_outer_dir(config_path)
    switched_config_path_container = []
    for i, fs in enumerate(switched_feature_selections_container):
        switched_config_file_name = f"{variant_name}_{SWITCHED_VARIANT_NAME_POSTFIX}{i + 1}.features"
        switched_config_file_path = join_path(configs_dir, switched_config_file_name)
        if not is_path_exist(switched_config_file_path):
            with open(switched_config_file_path, "w+") as output_file:
                for f_index, name in enumerate(fs):
                    line = name if fs[name] else f"#{name}"
                    if f_index < len(fs) - 1:
                        line += "\n"
                    output_file.write(line)
        switched_config_path_container.append(switched_config_file_path)
    return switched_config_path_container


def get_single_switched_feature_selections(feature_selections):
    feature_selections_container = []
    for name, status in feature_selections.items():
        switched_status = not status
        switched_feature_selections = dict(feature_selections)
        switched_feature_selections[name] = switched_status
        feature_selections_container.append(switched_feature_selections)
    return feature_selections_container


def get_feature_selections(config_path):
    feature_selections = {}
    with open(config_path) as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    feature_name = line[1:]
                    status = False
                else:
                    feature_name = line
                    status = True
                feature_selections[feature_name] = status
    return feature_selections


def compose_switched_products(config_paths, project_dir, mutated_project_dir, custom_ant):
    lib_paths = get_dependency_lib_dirs(project_dir)
    for config_path in config_paths:
        variant_dir = get_variant_dir_from_config_path(project_dir, config_path)
        corrupt_file = join_path(variant_dir, "corrupted_compile.log")
        if not is_path_exist(variant_dir):
            variant_dir = VariantComposer.compose_by_config(project_dir, config_path)
            try:
                compile_log = AntManager.compile_source_classes(lib_paths=lib_paths, variant_dir=variant_dir)
            except RuntimeError:
                print("********\n__FAILED__", config_path, "\n********\n")
                touch_file(corrupt_file)
                # delete_dir(variant_dir)
                continue
            else:
                TestManager.generate_junit_test_cases(lib_paths=lib_paths, variant_dir=variant_dir)
                TestManager.run_batch_junit_test_cases(variant_dir, lib_paths=lib_paths, halt_on_failure=True,
                                                       halt_on_error=True, custom_ant=custom_ant)
        elif is_path_exist(corrupt_file):
            print("********\n\n__SKIP__FAILED__", config_path, "\n********\n")
            continue

        mutated_variant_dir = get_variant_dir_from_config_path(mutated_project_dir, config_path)
        if not is_path_exist(mutated_variant_dir):
            mutated_variant_dir = VariantComposer.compose_by_config(mutated_project_dir, config_path)
            TestManager.link_generated_junit_test_cases(variant_dir, mutated_variant_dir)
            try:
                are_all_tests_passed = TestManager.run_batch_junit_test_cases(mutated_variant_dir, lib_paths=lib_paths,
                                                                              halt_on_failure=False,
                                                                              halt_on_error=True, custom_ant=custom_ant)
            except Exception as e:
                continue
            if are_all_tests_passed:
                file_name = SW_PASSED_TEST_FLAG_FILE_NAME
            else:
                file_name = SW_FAILED_TEST_FLAG_FILE_NAME
            test_flag_file = join_path(mutated_variant_dir, file_name)
            touch_file(test_flag_file)


def summarize_involving_features(project_dir, mutated_project_dir):
    variant_dirs = get_all_variant_dirs(mutated_project_dir, sort=True)
    features = get_ordered_feature_list(project_dir)
    buggy_features = get_mutated_features(mutated_project_dir)
    involving_features = set()
    for variant_dir in variant_dirs:
        variant_name = get_file_name(variant_dir)
        if is_path_exist(join_path(variant_dir, SW_PASSED_TEST_FLAG_FILE_NAME)):
            variant_postfix = variant_name.rsplit("_", 1)[1]
            if not variant_postfix.startswith(SWITCHED_VARIANT_NAME_POSTFIX):
                raise Exception("Invalid variant was considered as switched {}".format(variant_dir))
            feature_index = int(variant_postfix[len(SWITCHED_VARIANT_NAME_POSTFIX):]) - 1
            related_feature = features[feature_index]
            involving_features.add(related_feature)

    are_buggy_features_contained = buggy_features.issubset(involving_features)
    involving_features = list(buggy_features | involving_features)
    print("-" * 20)
    print("MUTATED_PROJECT_DIR:", mutated_project_dir)
    print("MUTANT NAME:", get_file_name(mutated_project_dir))
    print("BUGGY FEATURES:", json.dumps(list(buggy_features)))
    print("INVOLVING FEATURES:", json.dumps(involving_features))
    print("SIZE: {}".format(len(involving_features)))
    print("CONTAINED |{}|".format(are_buggy_features_contained))
    print("-" * 20)


def get_ordered_feature_list(project_dir):
    feature_order_file_path = get_feature_order_file_path(project_dir)
    features = []
    with open(feature_order_file_path) as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                features.append(line)
    return features
