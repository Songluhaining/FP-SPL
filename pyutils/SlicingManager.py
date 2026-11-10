import logging
import time

from util.FileManager import get_plugin_path, get_file_name_with_parent, get_slicing_log_file_path, get_outer_dir, \
    get_spectrum_failed_coverage_file_name_with_version, is_path_exist
from util.Helpers import get_logger, execute_shell_command

logger = get_logger(__name__)

PLUGIN_NAME = "feature-slicing.jar"
PLUGIN_PATH = get_plugin_path(PLUGIN_NAME)


def do_slice(spc_file_path, filtering_coverage_rate, coverage_version):

    start_time = time.time()
    failed_coverage_file_name = get_spectrum_failed_coverage_file_name_with_version(version=coverage_version)   #spectrum_failed_coverage.xml
    if coverage_version != "":
        post_fix = str(filtering_coverage_rate) + "_" + coverage_version + "_"
    else:
        post_fix = filtering_coverage_rate
    slicing_output_path = get_slicing_log_file_path(get_outer_dir(spc_file_path), post_fix)
    if is_path_exist(slicing_output_path):
        logger.info(f"Used Old Slicing log file [{slicing_output_path}]")
        return slicing_output_path, 0

    logger.info(f"Running suspicious_statements_manager from spc file [{get_file_name_with_parent(spc_file_path)}]")
    output_log = execute_shell_command(
        f'java -Xmx256m -Dspc_path={spc_file_path} -Dslicing_output_path={slicing_output_path} -Dcoverage_file_name={failed_coverage_file_name} -jar {PLUGIN_PATH} ',
        extra_args=[], log_to_file=True)
    logger.info(f"Wrote suspicious_statements_manager output to file [{get_file_name_with_parent(slicing_output_path)}]")
    logging.info("[Runtime] suspicious_statements_manager %s: %s", slicing_output_path, time.time() - start_time)
    slicing_runtime = time.time() - start_time
    return slicing_runtime