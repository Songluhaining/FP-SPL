import csv
import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from methods.mutual_information import su_calculation
from util.FileManager import get_model_configs_report_path, get_variants_dir, join_path, get_src_dir, get_spc_log_file_path, \
    get_file_name_with_parent, is_path_exist
from util.Helpers import get_logger, powerset
from util.TestingCoverageManager import statement_coverage_of_variants

logger = get_logger(__name__)


def find_SPCs(system, mutated_project_dir, filtering_coverage_rate):      #SPCs = Suspicious Partial Configurations
    start_time = time.time()
    spc_log_file_path = get_spc_log_file_path(mutated_project_dir, filtering_coverage_rate)   #spc_{}.log
    if is_path_exist(spc_log_file_path):
        logger.info(f"Used Old SPC log file [{spc_log_file_path}]")
        return spc_log_file_path, 0

    config_report_path = get_model_configs_report_path(mutated_project_dir)
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_testing_coverage = statement_coverage_of_variants(mutated_project_dir)
    feature_names, variant_names, passed_configs, failed_configs = load_configs(config_report_path,
                                                                                variants_testing_coverage,
                                                                                filtering_coverage_rate)
    df = pd.read_csv(config_report_path)
    df_data = df.iloc[:, 1:]
    columns = df_data.columns
    df_data.replace({'  T  ': 1, '  F  ': 0}, inplace=True)
    df_data.replace({'__PASSED__': 1, '__FAILED__': 0}, inplace=True)
    x = df_data[columns[:-1]]
    y = df_data[columns[-1]]

    dim = len(columns) - 1
    columns_name = x.columns
    C_relevance_dict = set()
    selected_featured_name = {}
    for i in range(0, dim):
        tem = su_calculation(x.values[:, i], y.values)
        selected_featured_name[columns_name[i]] = tem
        if tem > 0:
            C_relevance_dict.add(f"{i}_{True}")
            C_relevance_dict.add(f"{i}_{False}")

    spc_log_file_path, total_counter, nway_spc_number, inclusion_rate, duplication_rate = detect_SPCs(system, feature_names, passed_configs, failed_configs, variant_names, variants_dir,
                                    spc_log_file_path, C_relevance_dict, df)
    #logging.info("[Runtime] SPC runtime %s: %s", mutated_project_dir, time.time() - start_time)
    spc_runtime = time.time() - start_time
    return spc_log_file_path, spc_runtime, total_counter, nway_spc_number, inclusion_rate, duplication_rate


def remove_subsets(input_list):
    # 创建一个空列表来存储结果
    result = []

    # 遍历输入列表中的每个集合
    for current_set in input_list:
        is_subset = False

        # 检查当前集合是否是其他集合的子集
        for other_set in input_list:
            if len(current_set) > len(other_set):
                continue
            if current_set != other_set and current_set.issubset(other_set):
                is_subset = True
                break

        # 如果当前集合不是任何其他集合的子集，则将其添加到结果列表中
        if not is_subset:
            result.append(current_set)

    return result

def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))

def detect_SPCs(system, feature_names, passed_configs, failed_configs, variant_names, variants_dir, spc_log_file_path, C_relevance_dict, df):
    SPC_set = []
    switches_list = []
    Cache_set = set()
    saved_counter=0
    total_counter=0
    inclusion_rate = 0
    fl = len(feature_names)
    if (len(passed_configs) == 0 or len(failed_configs) == 0):
        spc_file = open(spc_log_file_path, "w")
        spc_file.close()
        return spc_log_file_path
    else:
        experience_dis = {"Email": 2.0, "Elevator": 2.0, "ExamDB": 2.5, "GPL": 2.5}
        logger.info(f"Finding SPCs and write to [{get_file_name_with_parent(spc_log_file_path)}]")
        with open(spc_log_file_path, "w+") as spc_log_file:
            # Core Algorithm
            for current_failed_config in failed_configs:
                switches = []
                current_failed_config_name = variant_names[tuple(current_failed_config)]
                position_current_failed_config = df.loc[df['Product\Feature']==current_failed_config_name]
                position_current_failed_config = position_current_failed_config.iloc[0, 2:].tolist()
                X = np.array(position_current_failed_config)
                X_with_each_failed_config_dis = {}
                avg_dis = 0
                for current_passed_config in passed_configs:
                    current_passed_config_name = variant_names[tuple(current_passed_config)]
                    position_current_passed_config = df.loc[df['Product\Feature'] == current_passed_config_name]
                    position_current_passed_config = position_current_passed_config.iloc[0, 2:].tolist()
                    Y = np.array(position_current_passed_config)
                    X_with_each_failed_config_dis[tuple(current_passed_config)] = eucliDist(X, Y)
                    avg_dis += X_with_each_failed_config_dis[tuple(current_passed_config)]
                X_with_each_failed_config_dis = sorted(X_with_each_failed_config_dis.items(), key=lambda x: x[1])
                X_with_each_failed_config_dis = dict(X_with_each_failed_config_dis)
                if system in experience_dis:
                    avg_dis = experience_dis[system]
                else:
                    avg_dis= round(avg_dis/len(X_with_each_failed_config_dis), 5)
                for current_passed_config in X_with_each_failed_config_dis:
                    if X_with_each_failed_config_dis[current_passed_config] < avg_dis:
                        current_switch = find_switched_feature_selections(current_failed_config,
                                                                      current_passed_config)
                        switches.append(current_switch)

                if len(switches) > 0:
                    switches = minimize_switches(switches)
                    switched_feature_selections = union_all_switched_feature_selections(switches)  # 各个switch的特征交集
                    switches_list.append(switched_feature_selections)
                else:
                    for current_passed_config in passed_configs:
                        current_switch = find_switched_feature_selections(current_failed_config,
                                                                          current_passed_config)
                        switches.append(current_switch)
                    if len(switches) > 0:
                        switches = minimize_switches(switches)
                        switched_feature_selections = union_all_switched_feature_selections(
                            switches)
                        switches_list.append(switched_feature_selections)
            cached_spc = []
            old_switches = switches_list
            switches_list = remove_subsets(switches_list)
            inclusion_rate = float((len(old_switches) - len(switches_list)) / len(old_switches))
            nway_spc_number = []
            for k in range(1, 8):
                inner_start = time.time()
                spc_number = 0
                for i, switched_feature_selected in enumerate(switches_list):

                    kth_set = set(combinations(switched_feature_selected, k))
                    for item in kth_set:
                        current_SPC = set(item)
                        total_counter += 1
                        if len(C_relevance_dict.intersection(current_SPC)) <= 0:
                            saved_counter += 1
                            continue
                        if tuple(current_SPC) in Cache_set:
                            saved_counter += 1
                            continue
                        spc_number += 1
                        Cache_set.add(tuple(current_SPC))
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
                                    spc_log_file.write(
                                        f"{combined_spc}; {get_src_dir(join_path(variants_dir, variant_names[tuple(spc_config)]))}\n")
                                cached_spc.append(combined_spc)
                            SPC_set.append(current_SPC)
                nway_spc_number.append(spc_number)
            if total_counter > 0:
                duplication_rate = (float(saved_counter / total_counter))
            else:
                duplication_rate = 0
                Cache_set = set()
            # with open("/home/whn/codes/VARCOP-gh-pages/SPC_set.txt", 'a') as file:
            #     file.write(str(SPC_set)+'\n')
            return spc_log_file_path, (total_counter - saved_counter), nway_spc_number, inclusion_rate, duplication_rate
            # end



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
    fs = eval(fs)   #还原fs为bool类型
    return feature_position, fs


def exist_configs_contain_spc(SPC, configs):
    has_configs_contain_spc = False
    for fc in configs:
        valid_fs = []
        for spc_fs in SPC:
            feature_position, config_fs = split_positioned_feature_selection(spc_fs)   #feature_position为数字，表示第几个特征；config_fs为bool类型，表示该特征的选择
            if fc[feature_position] == config_fs:
                valid_fs.append(True)
        if len(valid_fs) == len(SPC):   #SPC为configs的子集
            has_configs_contain_spc = True
            break

    return has_configs_contain_spc


def union_all_switched_feature_selections(switches):
    switched_feature_selections = switches[0].union(*switches[1:])      #union并集  intersection交集  difference差集
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
            feature_selection = f"{feature_position}_{failed_config_fs}"
            switched_feature_selections.add(feature_selection)


    return switched_feature_selections  # ['1_False', '2_True', '7_True', ...]


def load_configs(config_report_path, variants_testing_coverage, filtering_coverage_rate):
    logger.info(f"Loading config report file [{get_file_name_with_parent(config_report_path)}]")
    with open(config_report_path) as f:      #一行一行的读
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
                #if variants_testing_coverage[current_variant_name] >= filtering_coverage_rate:
                passed_configs.append(current_config)
                variant_names[tuple(current_config)] = current_variant_name
            else:
                failed_configs.append(current_config)
                variant_names[tuple(current_config)] = current_variant_name
        return feature_names, variant_names, passed_configs, failed_configs

