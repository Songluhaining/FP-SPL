from FileManager import get_plugin_path, is_path_exist, get_feature_order_file_path, get_file_name, get_model_file_path, \
    get_implemented_features
from Helpers import get_logger, execute_shell_command

logger = get_logger(__name__)

PLUGIN_NAME = "FeatureOrderTool.jar"
PLUGIN_PATH = get_plugin_path(PLUGIN_NAME)


def generate_feature_order_file(project_dir):
    logger.info(f"Generating Feature Order file from model file at project [{get_file_name(project_dir)}]")
    model_file_path = get_model_file_path(project_dir)
    implemented_features = get_implemented_features(project_dir)
    output_feature_order_path = get_feature_order_file_path(project_dir)
    if not is_path_exist(output_feature_order_path):
        execute_shell_command(f'java -jar {PLUGIN_PATH} ', extra_args=[
            {"--feature_model": model_file_path},
            {"--available_features": ",".join(implemented_features)},
            {"--output_feature_order_path": output_feature_order_path},
        ])
    else:
        logger.info(f"Used Custom Feature Order file")

    return output_feature_order_path


def read_feature_order_file(file_path):
    with open(file_path) as input_file:
        feature_list = [line.strip() for line in input_file.readlines()]
        return feature_list
