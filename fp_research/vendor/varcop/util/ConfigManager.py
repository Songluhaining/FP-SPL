import csv
import re
from collections import defaultdict, OrderedDict

from FileManager import get_model_configs_dir, get_file_name_without_ext, join_path, list_dir, \
    get_model_configs_report_path, \
    copy_file
from Helpers import get_logger
from util import ModelManager

logger = get_logger(__name__)


def get_optional_feature_names(sampling_file_path):
    optional_feature_names = []
    with open(sampling_file_path, "r") as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            label = re.sub("[\[\]\* ]", '', row[0])
            feature_statuses = row[2:-1]
            is_optional_feature = -1 < len(list(filter(lambda status: status == "X", feature_statuses)))
            if is_optional_feature:
                optional_feature_names.append(label)
    return optional_feature_names


def get_configurations_from_sampling_file(ordered_features, sampling_file_path):
    with open(sampling_file_path, "r") as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        next(csv_reader)
        configurations = defaultdict(OrderedDict)
        sampling_matrix = list(filter(lambda x: x[0] in ordered_features, list(csv_reader)))
        sampling_matrix.sort(key=lambda row: ordered_features.index(row[0]))
        for row in sampling_matrix:
            label = re.sub("[\[\]\* ]", '', row[0])
            for i, feature_selection in enumerate(row[1:-1]):
                configurations[i][label] = feature_selection == "X"
    return list(configurations.values())


def write_configuration_to_file(configuration, output_file_path):
    lines = []
    has_enabled_feature = False
    for feature, is_enabled in configuration.items():
        # comment all disabled features
        line = ("" if is_enabled else "#") + feature
        if is_enabled and not has_enabled_feature:
            has_enabled_feature = True
        lines.append(line)

    if not has_enabled_feature:
        return False
    with open(output_file_path, "w+") as output_file:
        output_file.write("\n".join(lines))
    return True


def generate_config_name(sampling_file_name, index):
    return f"{sampling_file_name}_{str(index + 1).zfill(4)}"


PRODUCT_FEATURE_FIELD = 'Product\Feature'
OUTPUT_FIELD = '__TEST_OUTPUT__'


def generate_configs_report(ordered_features, config_output_paths, configurations, configs_report_output_path):
    assert (isinstance(ordered_features, list) and isinstance(config_output_paths, list) and isinstance(configurations,
                                                                                                        list))

    with open(configs_report_output_path, 'w') as output_csv:
        field_names = [PRODUCT_FEATURE_FIELD] + ordered_features + [OUTPUT_FIELD]
        writer = csv.DictWriter(output_csv, fieldnames=field_names)

        writer.writeheader()
        for config_output_path, config in zip(config_output_paths, configurations):
            config_name = get_file_name_without_ext(config_output_path)
            for feature in config:
                is_feature_enabled = config[feature]
                if is_feature_enabled:
                    flag = "T"
                else:
                    flag = "F"
                config[feature] = '{:^5}'.format(flag)

            config[PRODUCT_FEATURE_FIELD] = config_name
            config[OUTPUT_FIELD] = "__NOASWR__"
            writer.writerow(config)


def generate_configs(project_dir, feature_order_file_path, sampling_output_file_path):
    logger.info(
        f"Generating configurations from sampling csv file [{get_file_name_without_ext(sampling_output_file_path)}]")
    config_dir = get_model_configs_dir(project_dir)
    sampling_file_name = get_file_name_without_ext(sampling_output_file_path).replace(".", "_")
    ordered_features = ModelManager.read_feature_order_file(feature_order_file_path)
    configurations = get_configurations_from_sampling_file(ordered_features, sampling_output_file_path)
    config_output_paths = []
    for i, config in enumerate(configurations[:]):
        config_name = generate_config_name(sampling_file_name, i)
        config_output_path = join_path(config_dir, f"{config_name}.features")
        success = write_configuration_to_file(config, config_output_path)
        if success:
            config_output_paths.append(config_output_path)
        else:
            configurations.remove(config)
    configs_report_file_path = get_model_configs_report_path(project_dir)
    generate_configs_report(ordered_features, config_output_paths, configurations, configs_report_file_path)
    return configs_report_file_path, config_output_paths


def get_config_paths(project_dir, sort=False):
    config_dir = get_model_configs_dir(project_dir)
    return list_dir(config_dir, full_path=True, sort=sort)


def copy_configs_report(src_project_dir, dst_project_dir):
    src_configs_report_file_path = get_model_configs_report_path(src_project_dir)
    dst_configs_report_file_path = get_model_configs_report_path(dst_project_dir)
    copy_file(src_configs_report_file_path, dst_configs_report_file_path)
