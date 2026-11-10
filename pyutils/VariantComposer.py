import re

from FileManager import get_plugin_path, get_variant_dir, get_feature_source_code_dir, move_file, \
    get_file_name_without_ext, join_path, is_path_exist, get_src_dir, get_file_name, get_variants_dir
from Helpers import get_logger, execute_shell_command

logger = get_logger(__name__)

PLUGIN_NAME = "featurehouse_20210219.jar"
PLUGIN_PATH = get_plugin_path(PLUGIN_NAME)


def get_sampling_file_path(stdout):
    return re.search("(?<=Wrote result to )[^\n]+", stdout).group()


def compose_by_config(project_dir, config_file_path):
    logger.info(
        f"Composing [{get_file_name(project_dir)}] project's source code with config file [{get_file_name_without_ext(config_file_path)}]")
    config_name = get_file_name_without_ext(config_file_path)
    output_dir = get_variant_dir(project_dir, config_name)
    execute_shell_command(f'java -jar {PLUGIN_PATH}', extra_args=[
        {"--expression": config_file_path},
        {"--base-directory": get_feature_source_code_dir(project_dir)},
        {"--output-directory": output_dir},
        {"--export_roles_json": ""},
        {"--featureAnnotationJava": ""}
    ])
    output_src_dir = join_path(output_dir, config_name)
    if is_path_exist(output_src_dir):
        renamed_folder_dir = get_src_dir(output_dir)
        move_file(output_src_dir, renamed_folder_dir)
    return output_dir


def was_variants_composed(project_dir):
    variants_dir = get_variants_dir(project_dir)
    return len(variants_dir) > 0
