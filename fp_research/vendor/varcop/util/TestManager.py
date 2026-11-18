import csv
import re

from util.FileManager import get_plugin_path, get_file_name_without_ext, get_test_dir, create_symlink, \
    get_compiled_source_classes_dir, \
    get_compiled_test_classes_dir, get_file_name, get_src_dir, get_test_coverage_dir, get_variant_dir, is_path_exist, \
    join_path, get_model_configs_report_path, delete_dir, get_variants_dir, list_dir, get_dependency_lib_dirs, \
    touch_file, FAILED_TEST_COVERAGE_FOLDER_NAME, PASSED_TEST_COVERAGE_FOLDER_NAME, move_file
from Helpers import get_logger, execute_shell_command, hash_md5

logger = get_logger(__name__)

EVOSUITE_PLUGIN_NAME = "evosuite-1.0.6.jar"
EVOSUITE_PLUGIN_PATH = get_plugin_path(EVOSUITE_PLUGIN_NAME)

JUNIT_PLUGIN_NAME = "es_junit.sh"
JUNIT_PLUGIN_PATH = get_plugin_path(JUNIT_PLUGIN_NAME)

BATCH_JUNIT_PLUGIN_NAME = "es_batch_junit.sh"
BATCH_JUNIT_PLUGIN_PATH = get_plugin_path(BATCH_JUNIT_PLUGIN_NAME)


def generate_junit_test_cases(lib_paths, variant_dir):
    logger.info(f"Generating JUnit Test for variant [{get_file_name_without_ext(variant_dir)}]")
    compiled_classes_dir = get_compiled_source_classes_dir(variant_dir)

    evosuite_temp_dir = join_path(".evosuite_" + hash_md5(variant_dir))
    delete_dir(evosuite_temp_dir)

    test_cases_dir = get_test_dir(variant_dir, force_mkdir=False)
    delete_dir(test_cases_dir)

    output_log = execute_shell_command(f'java -jar {EVOSUITE_PLUGIN_PATH}', extra_args=[
        {"-projectCP": ":".join([compiled_classes_dir] + lib_paths)},
        {"-seed": 1583738192420},
        {"-target": compiled_classes_dir},
        {"-continuous": "execute"},
        {"-Dctg_memory": "4000"},
        {"-Dctg_cores": "4"},
        {"-Dctg_dir": evosuite_temp_dir},
        {"-Dctg_export_folder": test_cases_dir},
    ], log_to_file=True)


def link_generated_junit_test_cases(variant_dir, target_variant_dir):
    logger.info(f"Linking JUnit Test for variant [{get_file_name_without_ext(variant_dir)}]")
    generated_test_dir = get_test_dir(variant_dir)
    target_test_dir = get_test_dir(target_variant_dir, force_mkdir=False)
    create_symlink(generated_test_dir, target_test_dir)


def run_batch_junit_test_cases(variant_dir, lib_paths=None, halt_on_failure=False, halt_on_error=True,
                               custom_ant=None):
    if lib_paths is None:
        lib_paths = []
    logger.info(
        f"Running [BATCH] JUnit Test for variant [{get_file_name_without_ext(variant_dir)}] - Using custom ant [{custom_ant}]")
    src_dir = get_src_dir(variant_dir)
    test_dir = get_test_dir(variant_dir)
    src_classes_dir = get_compiled_source_classes_dir(variant_dir)
    test_classes_dir = get_compiled_test_classes_dir(variant_dir)
    output_log = execute_shell_command(f'/bin/sh {BATCH_JUNIT_PLUGIN_PATH}', extra_args=[
        {"-src": src_dir},
        {"-test": test_dir},
        {"-build.classes": src_classes_dir},
        {"-build.testclasses": test_classes_dir},
        {"-report.coveragedir": "_"},
        {"-junit.haltonfailure": "yes" if halt_on_failure else "no"},
        {"-ant.name": custom_ant},
        {"-lib_path": ":".join(lib_paths)},
    ], log_to_file=True)
    is_test_failure = re.search("(Failures: [1-9]+|Errors: [1-9]+|BUILD FAILED)", output_log)
    if is_test_failure:
        if halt_on_failure or (halt_on_error and "BUILD FAILED" in is_test_failure.group()):
            logger.fatal("Some test cases were failed, see log for more detail\n{}".format(output_log))
            raise RuntimeError("Test case failures")
        return False
    return True


def run_junit_test_cases_with_coverage(variant_dir, lib_paths=None, halt_on_failure=False, halt_on_error=True,
                                       custom_ant=None):
    if lib_paths is None:
        lib_paths = []
    logger.info(
        f"Running JUnit Test for variant [{get_file_name_without_ext(variant_dir)}] - Using custom ant [{custom_ant}]")
    src_dir = get_src_dir(variant_dir)
    test_dir = get_test_dir(variant_dir)
    src_classes_dir = get_compiled_source_classes_dir(variant_dir)
    test_classes_dir = get_compiled_test_classes_dir(variant_dir)
    test_coverage_dir = get_test_coverage_dir(variant_dir)
    output_log = execute_shell_command(f'/bin/sh {JUNIT_PLUGIN_PATH}', extra_args=[
        {"-src": src_dir},
        {"-test": test_dir},
        {"-build.classes": src_classes_dir},
        {"-build.testclasses": test_classes_dir},
        {"-report.coveragedir": test_coverage_dir},
        {"-junit.haltonfailure": "yes" if halt_on_failure else "no"},
        {"-ant.name": custom_ant},
        {"-lib_path": ":".join(lib_paths)},
    ], log_to_file=True)

    is_test_failure = re.search("(Failures: [1-9]+|Errors: [1-9]+|BUILD FAILED)", output_log)
    if is_test_failure:
        if halt_on_failure or (halt_on_error and "BUILD FAILED" in is_test_failure.group()):
            logger.fatal("Some test cases were failed, see log for more detail\n{}".format(output_log))
            raise RuntimeError("Test case failures")
        return False
    return True


def run_junit_test_cases_with_coverage_on_project(project_dir, custom_ant=None):
    variants_dir = get_variants_dir(project_dir)
    lib_paths = get_dependency_lib_dirs(project_dir)
    for variant_dir in list_dir(variants_dir, full_path=True):
        run_junit_test_cases_with_coverage(variant_dir=variant_dir, lib_paths=lib_paths, custom_ant=custom_ant)


PASSED_TEST_FLAG_FILE_NAME = "batch.test.passed.flag"
FAILED_TEST_FLAG_FILE_NAME = "batch.test.failed.flag"
UNKNOWN_TEST_FLAG_FILE_NAME = "batch.test.unknown.flag"


class JunitMode:
    FAST = 1
    FULL_COVERAGE = 2


def run_junit_test_cases_on_project(project_dir, junit_mode=JunitMode.FAST, custom_ant=None):
    if junit_mode == JunitMode.FAST:
        run_batch_junit_test_cases_on_project(project_dir, custom_ant=custom_ant)
    elif junit_mode == JunitMode.FULL_COVERAGE:
        run_junit_test_cases_with_coverage_on_project(project_dir, custom_ant=custom_ant)
    else:
        raise Exception("Invalid junit mode")


def run_batch_junit_test_cases_on_project(project_dir, custom_ant=None, custom_variant_dirs=None):
    if not custom_variant_dirs:
        variants_dir = get_variants_dir(project_dir)
        custom_variant_dirs = list_dir(variants_dir, full_path=True)
    lib_paths = get_dependency_lib_dirs(project_dir)

    for variant_dir in custom_variant_dirs:
        are_all_tests_passed = run_batch_junit_test_cases(variant_dir=variant_dir, lib_paths=lib_paths,
                                                          custom_ant=custom_ant)
        if are_all_tests_passed is True:
            file_name = PASSED_TEST_FLAG_FILE_NAME
        elif are_all_tests_passed is False:
            file_name = FAILED_TEST_FLAG_FILE_NAME
        else:
            logger.warning(f"Invalid are_all_tests_passed value {are_all_tests_passed} on variant {variant_dir}")
            file_name = FAILED_TEST_FLAG_FILE_NAME
        test_flag_file = join_path(variant_dir, file_name)
        touch_file(test_flag_file)


def check_variant_final_test_output(variant_dir, junit_mode=JunitMode.FAST):
    coverage_dir = get_test_coverage_dir(variant_dir)
    if is_path_exist(join_path(variant_dir, FAILED_TEST_FLAG_FILE_NAME)):
        if junit_mode == JunitMode.FULL_COVERAGE and not is_path_exist(
                join_path(coverage_dir, FAILED_TEST_COVERAGE_FOLDER_NAME)):
            return None
        return False
    elif is_path_exist(join_path(variant_dir, PASSED_TEST_FLAG_FILE_NAME)):
        if junit_mode == JunitMode.FULL_COVERAGE and not is_path_exist(
                join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME)):
            return None
        return True
    else:
        return None


def write_test_output_to_configs_report(project_dir, junit_mode=JunitMode.FAST):
    logger.info(f"Writing test output to project's configs report [{get_file_name(project_dir)}]")
    configs_report_file_path = get_model_configs_report_path(project_dir)
    rows = []
    with open(configs_report_file_path, "r") as report_csv:
        reader = csv.reader(report_csv)
        for i, row in enumerate(reader):
            rows.append(row)
            if i == 0:
                continue
            config_name = row[0]
            variant_dir = get_variant_dir(project_dir, config_name)
            final_test_output_flag = check_variant_final_test_output(variant_dir, junit_mode=junit_mode)
            if final_test_output_flag is True:
                final_test_output = "__PASSED__"
            elif final_test_output_flag is False:
                final_test_output = "__FAILED__"
            else:
                raise Exception("Overall test result is __NOASWR__ in variant {}".format(variant_dir))
            row[-1] = final_test_output
    with open(configs_report_file_path, "w") as output_report_csv:
        writer = csv.writer(output_report_csv)
        writer.writerows(rows)
    if JunitMode.FULL_COVERAGE:
        move_file(configs_report_file_path, configs_report_file_path + ".done")


def get_failed_variant_dirs_from_config_report(mutated_project_dir):
    configs_report_file_path = get_model_configs_report_path(mutated_project_dir)
    variants_dir = get_variants_dir(mutated_project_dir)
    variant_dir_dict = {get_file_name(vd): vd for vd in list_dir(variants_dir, full_path=True)}
    failed_variant_dirs = []
    with open(configs_report_file_path, "r") as report_csv:
        reader = csv.reader(report_csv)
        next(reader)
        for i, row in enumerate(reader):
            test_passed = row[-1] == "__PASSED__"
            current_variant_name = row[0]
            if not test_passed:
                failed_variant_dirs.append(variant_dir_dict.get(current_variant_name))
    return failed_variant_dirs
