import csv
import time

from util.FileManager import get_model_configs_report_path, get_variants_dir, join_path, get_src_dir, get_spc_log_file_path, \
    get_file_name_with_parent, is_path_exist
from util.Helpers import get_logger, powerset
from util.TestingCoverageManager import statement_coverage_of_variants

logger = get_logger(__name__)


def find_SPCs(mutated_project_dir, filtering_coverage_rate):
    start_time = time.time()
    spc_log_file_path = get_spc_log_file_path(mutated_project_dir, filtering_coverage_rate)
    if is_path_exist(spc_log_file_path):
        logger.info(f"Used Old SPC log file [{spc_log_file_path}]")
        return spc_log_file_path, 0

    config_report_path = get_model_configs_report_path(mutated_project_dir)
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_testing_coverage = statement_coverage_of_variants(mutated_project_dir)
    feature_names, variant_names, passed_configs, failed_configs = load_configs(config_report_path,
                                                                                variants_testing_coverage,
                                                                                filtering_coverage_rate)
    spc_log_file_path = detect_SPCs(feature_names, passed_configs, failed_configs, variant_names, variants_dir,
                                    spc_log_file_path)
    #logging.info("[Runtime] SPC runtime %s: %s", mutated_project_dir, time.time() - start_time)
    spc_runtime = time.time() - start_time
    return spc_log_file_path, spc_runtime


def detect_SPCs(feature_names, passed_configs, failed_configs, variant_names, variants_dir, spc_log_file_path):
    SPC_set = []
    if (len(passed_configs) == 0 or len(failed_configs) == 0):
        spc_file = open(spc_log_file_path, "w")
        spc_file.close()
        return spc_log_file_path
    else:
        logger.info(f"Finding SPCs and write to [{get_file_name_with_parent(spc_log_file_path)}]")
        with open(spc_log_file_path, "w+") as spc_log_file:
            for current_failed_config in failed_configs:
                switches = []
                for current_passed_config in passed_configs:
                    current_switch = find_switched_feature_selections(current_failed_config, current_passed_config)
                    switches.append(current_switch)
                switches = minimize_switches(switches)
                switched_feature_selections = union_all_switched_feature_selections(switches)
                cached_spc = []
                for current_SPC in powerset(switched_feature_selections):
                    if (len(current_SPC) > 7):
                        break
                    if len(current_SPC) <= 0:
                        continue
                    current_SPC = set(current_SPC)
                    if satisfy_spc_minimality(current_SPC, SPC_set) and satisfy_spc_necessity(current_SPC,
                                                                                              passed_configs,
                                                                                              failed_configs):
                        combined_spc = combine_spc_with_feature_names(feature_names, current_SPC)
                        if combined_spc.strip() and combined_spc not in cached_spc:
                            # minimized_failed_config = find_minimized_failed_config_contains_spc(current_SPC,
                            #                                                                     failed_configs)
                            spc_failed_configs = find_failed_configs_contains_spc(current_SPC,
                                                                                                failed_configs)
                            for spc_config in spc_failed_configs:
                                spc_log_file.write(f"{combined_spc}; {get_src_dir(join_path(variants_dir, variant_names[tuple(spc_config)]))}\n")
                            # print()
                            # print(f"{combined_spc}")
                            cached_spc.append(combined_spc)
                        SPC_set.append(current_SPC)
            # final_SPC_set = combine_spc_set_with_feature_names(feature_names, SPC_set)
            # print(final_SPC_set)
            return spc_log_file_path


def find_minimized_failed_config_contains_spc(current_SPC, failed_configs):
    minimized_failed_config = None
    min_enabled_fs_count = 10000
    for fc in failed_configs:
        valid_fs = []
        for spc_fs in current_SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(current_SPC):
            current_enabled_fs_count = fc.count(True)
            if current_enabled_fs_count < min_enabled_fs_count:
                min_enabled_fs_count = current_enabled_fs_count
                minimized_failed_config = fc
    if not minimized_failed_config:
        raise Exception("Not found any failed config contains SPC {}".format(current_SPC))
    return minimized_failed_config


def find_failed_configs_contains_spc(current_SPC, failed_configs):
    needing_failed_configs = []
    for fc in failed_configs:
        valid_fs = []
        for spc_fs in current_SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(current_SPC):
            needing_failed_configs.append(fc)

    return needing_failed_configs


def combine_spc_set_with_feature_names(feature_names, SPC_set):
    combined_SPC_set = []
    for current_SPC in SPC_set:
        new_SPC = combine_spc_with_feature_names(feature_names, current_SPC)
        combined_SPC_set.append(new_SPC)
    return combined_SPC_set


def combine_spc_with_feature_names(feature_names, current_SPC):
    new_SPC = {}
    for spc_fs in current_SPC:
        feature_position, fs = split_positioned_feature_selection(spc_fs)
        current_feature_name = feature_names[feature_position]
        if fs is True:
            new_SPC[current_feature_name] = fs
        else:
            new_SPC["#" + current_feature_name] = fs
    return ",".join(new_SPC.keys())


def satisfy_spc_minimality(current_SPC, SPC_set):
    for added_SPC in SPC_set:
        if is_child_switch(added_SPC, current_SPC):
            return False
    return True


def satisfy_spc_necessity(SPC, passed_configs, failed_configs):
    return exist_configs_contain_spc(SPC, failed_configs) and not exist_configs_contain_spc(SPC, passed_configs)


def split_positioned_feature_selection(positioned_fs):
    feature_position, fs = positioned_fs.split("_", 1)
    feature_position = int(feature_position)
    fs = eval(fs)
    return feature_position, fs


def exist_configs_contain_spc(SPC, configs):
    has_configs_contain_spc = False
    for fc in configs:
        valid_fs = []
        for spc_fs in SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(SPC):
            has_configs_contain_spc = True
            break

    return has_configs_contain_spc


def union_all_switched_feature_selections(switches):
    switched_feature_selections = switches[0].union(*switches[1:])
    return switched_feature_selections


def minimize_switches(switches):
    switches.sort(key=lambda s: len(s), reverse=False)
    for i, current_switch in enumerate(switches):
        for target_switch in switches[i + 1:]:
            if is_child_switch(current_switch, target_switch):
                switches.remove(target_switch)
    return switches


def is_child_switch(switch, target_switch):
    if len(switch.intersection(target_switch)) == len(switch):
        return True
    return False


def find_switched_feature_selections(failed_config, passed_config):
    switched_feature_selections = set()
    for feature_position, (failed_config_fs, passed_config_fs) in enumerate(zip(failed_config, passed_config)):
        if failed_config_fs != passed_config_fs:
            switched_feature_selections.add(f"{feature_position}_{failed_config_fs}")
    return switched_feature_selections  # ['1_False', '2_True', '7_True', ...]


def load_configs(config_report_path, variants_testing_coverage, filtering_coverage_rate):
    logger.info(f"Loading config report file [{get_file_name_with_parent(config_report_path)}]")
    with open(config_report_path) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        feature_names = header[1:]
        variant_names = {}
        passed_configs = []
        failed_configs = []
        for row in reader:
            current_variant_name, current_config, current_test_result = row[0], row[1:-1], row[-1]
            current_config = list(map(lambda fs: fs.strip() == "T", current_config))
            if current_test_result == "__NOASWR__":
                logger.fatal(f"Found untested variant [{current_variant_name}]")
            elif current_test_result == "__PASSED__":
                if variants_testing_coverage[current_variant_name] >= filtering_coverage_rate:
                    passed_configs.append(current_config)
                    variant_names[tuple(current_config)] = current_variant_name
            else:
                failed_configs.append(current_config)
                variant_names[tuple(current_config)] = current_variant_name
        return feature_names, variant_names, passed_configs, failed_configs
