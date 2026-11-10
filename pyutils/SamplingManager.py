import re

from FileManager import get_plugin_path, get_file_name_without_ext
from Helpers import get_logger, execute_shell_command

logger = get_logger(__name__)

PLUGIN_NAME = "SPLCATool.jar"
PLUGIN_PATH = get_plugin_path(PLUGIN_NAME)


def get_sampling_file_path(output_log):
    return re.search("(?<=Wrote result to ).+", output_log).group()


def sampling(model_file_path, t_wise):
    logger.info(f"Running sampling for model file [{get_file_name_without_ext(model_file_path)}] with {t_wise}-wise")
    output_log = execute_shell_command(f'java -jar {PLUGIN_PATH} ', extra_args=[
        {"-fm": model_file_path},
        {"-t": "t_wise"},
        {"-a": "Chvatal"},
        {"-s": t_wise},
    ])
    return get_sampling_file_path(output_log)
