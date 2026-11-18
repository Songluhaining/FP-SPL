import csv
import random
from collections import defaultdict
from itertools import combinations, product

from util.AntManager import check_all_variant_compilable
from FileManager import get_plugin_path, split_path, get_mutation_result_dir, list_dir, join_path, \
    get_mutated_projects_dir, create_symlink, get_feature_source_code_dir, get_file_name_without_ext, copy_dir, \
    is_path_exist, get_model_configs_report_path, get_project_name, unlink, is_symlink
from Helpers import get_logger, execute_shell_command, natural_sort
from suspicious_statements_manager.SuspiciousStatementManager import get_multiple_buggy_statements

logger = get_logger(__name__)

PLUGIN_NAME = "muJava.jar"
PLUGIN_PATH = get_plugin_path(PLUGIN_NAME)


def assign_current_project_as_new_session(current_project_dir):
    projects_dir, project_name = split_path(current_project_dir)
    execute_shell_command(f'java -Dmujava_home={projects_dir} -cp {PLUGIN_PATH} mujava.cli.testnew', extra_args=[
        {project_name: ""},
    ])


def get_all_mutant_paths(mutation_result_dir):
    """
        Mutation result tracing path for each mutated operator
            mutation_result -> feature_class (eg. Base.GPL.EdgeIter)
            -> traditional_mutants -> mutated method (eg. boolean_hasNext())
            -> mutation operators (eg. SDL_1) -> mutated class file (eg. EdgeIter.java)
        """

    output = execute_shell_command(f"find {mutation_result_dir}",
                                   extra_args=[
                                       {"-mindepth": 4},
                                       {"-type": "f"},
                                       {"-name": "\"*.java\""},
                                   ])
    return output.split("\n")[:-1]


def count_mutants(mutant_paths):
    return len(mutant_paths)


def make_mutants(current_project_dir, optional_feature_names):
    projects_dir, project_name = split_path(current_project_dir)
    execute_shell_command(
        f'java -Dmujava_home={projects_dir} -Dallowed_features={",".join(optional_feature_names)} -cp {PLUGIN_PATH} mujava.cli.genmutes',
        extra_args=[
            {"-all": ""},
            {project_name: ""},
        ])
    mutation_result_dir = get_mutation_result_dir(current_project_dir)
    mutant_paths = get_all_mutant_paths(mutation_result_dir)
    mutant_count = count_mutants(mutant_paths)
    logger.info(f"Composed {mutant_count} mutants [{get_file_name_without_ext(mutation_result_dir)}]")
    return mutant_paths


def generate_mutants(project_dir, optional_feature_names, num_of_bugs=1):
    logger.info(f"Mutating features [{get_file_name_without_ext(project_dir)}]")
    assign_current_project_as_new_session(project_dir)
    mutant_paths = make_mutants(project_dir, optional_feature_names)
    mutant_path_tuples = mixing_multiple_bugs(mutant_paths, num_of_bugs=num_of_bugs)
    mutated_project_dirs = inject_mutants(project_dir, mutant_path_tuples)
    return mutated_project_dirs


def regenerate_filtered_mutants_from_bug_tuples(project_dir, bug_tuples):
    logger.info(f"Remaking mutants [{get_file_name_without_ext(project_dir)}] from {len(bug_tuples)} bug TUPLEs")
    mutant_path_tuples = convert_bug_tuples_to_mutant_path_tuples(project_dir, bug_tuples)
    filtered_mutated_project_dirs = inject_mutants(project_dir, mutant_path_tuples)
    return filtered_mutated_project_dirs


def convert_bug_tuples_to_mutant_path_tuples(project_dir, bug_tuples):
    mutation_result_dir = get_mutation_result_dir(project_dir)
    mutant_paths = get_all_mutant_paths(mutation_result_dir)
    mutant_path_mapping = {}
    for mutant_path in mutant_paths:
        mutant_path_parts = mutant_path.rsplit("/", 5)
        full_class_name = mutant_path_parts[1]
        operator_index = mutant_path_parts[4]
        current_bug_id = f"{full_class_name}.{operator_index}"
        mutant_path_mapping[current_bug_id] = mutant_path
    mutant_path_tuples = []
    for bt in bug_tuples:
        mutant_path_tuples.append(tuple(mutant_path_mapping[bug_id] for bug_id in bt))
    return mutant_path_tuples


def regenerate_filtered_mutants_from_bug_ids(project_dir, bug_ids, num_of_bugs, case_limit):
    logger.info(f"Remaking mutants [{get_file_name_without_ext(project_dir)}] from {len(bug_ids)} bug IDs")
    filtered_mutant_paths = filter_mutants(project_dir, bug_ids)
    filtered_mutant_path_tuples = mixing_multiple_bugs(filtered_mutant_paths, num_of_bugs=num_of_bugs,
                                                       case_limit=case_limit, allow_same_file=True)
    filtered_mutated_project_dirs = inject_mutants(project_dir, filtered_mutant_path_tuples)
    return filtered_mutated_project_dirs


def filter_mutants(project_dir, bug_ids):
    mutation_result_dir = get_mutation_result_dir(project_dir)
    mutant_paths = get_all_mutant_paths(mutation_result_dir)
    filtered_mutant_paths = []
    for mutant_path in mutant_paths:
        mutant_path_parts = mutant_path.rsplit("/", 5)
        full_class_name = mutant_path_parts[1]
        operator_index = mutant_path_parts[4]
        current_bug_id = f"{full_class_name}.{operator_index}"
        if len(bug_ids) <= 0 or current_bug_id in bug_ids:
            filtered_mutant_paths.append(mutant_path)
    return filtered_mutant_paths


def mixing_multiple_bugs(mutant_paths, num_of_bugs=1, case_limit=None, allow_same_file=False):
    logger.info(f"Mixing multiple bugs [{num_of_bugs}] from {len(mutant_paths)} mutant files")
    if num_of_bugs == 0:
        return []
    elif num_of_bugs == 1:
        return [(source_file_path,) for source_file_path in mutant_paths]
    else:
        deduplicated_bug_dict = defaultdict(list)
        for source_file_path in mutant_paths:
            mutant_path_parts = source_file_path.rsplit("/", 5)
            full_class_name = mutant_path_parts[1]
            if not allow_same_file:
                if len(deduplicated_bug_dict[full_class_name]) > 0:
                    continue
            deduplicated_bug_dict[full_class_name].append(source_file_path)

        # algorithm (ex. 2 bugs)
        # original_list = [(1,2,3), (4,5), (6,7)]
        # because number of bugs is 2 so need to find all combinations of 2 sub lists first (by combinations func)
        # with each 2 sub list, we find all combinations of two elements that are picked from both of the lists ((by product func)

        candidate_combinations = list(combinations(deduplicated_bug_dict.values(), num_of_bugs))
        mixed_mutant_path_tuples = []
        for cc in candidate_combinations:
            current_mixing_tuple = list(product(*cc))
            mixed_mutant_path_tuples.extend(current_mixing_tuple)
        if 0 < case_limit < len(mixed_mutant_path_tuples):
            mixed_mutant_path_tuples = random.sample(mixed_mutant_path_tuples, k=case_limit)
        return mixed_mutant_path_tuples


def inject_mutants(project_dir, mutant_path_tuples):
    logger.info(f"Injecting mutants to features [{get_file_name_without_ext(project_dir)}]")
    mutated_projects_dir = get_mutated_projects_dir(project_dir)
    features_dir = get_feature_source_code_dir(project_dir)
    mutated_project_dirs = []
    for tuple_index, mutant_path_tuple in enumerate(mutant_path_tuples):
        # FORMAT: ~/InputPreparation/projects/GPL-Test/mutation_result/DirectedWithEdges.GPL.Edge/traditional_mutants/Vertex_getOtherVertex(Vertex)/ROR_1/Edge.java'
        current_project_name = "_MultipleBugs_.NOB_{}.ID_{}".format(len(mutant_path_tuple), tuple_index + 1)
        current_mutated_project_dir = join_path(mutated_projects_dir, current_project_name)
        current_mutated_features_dir = get_feature_source_code_dir(project_dir=current_mutated_project_dir)
        final_mutation_log = ""
        for mutant_path in mutant_path_tuple:
            mutant_path_parts = mutant_path.rsplit("/", 5)
            full_class_name = mutant_path_parts[1]
            package, class_name = full_class_name.rsplit(".", 1)
            class_name += ".java"
            operator_index = mutant_path_parts[4]
            bug_name = f"{full_class_name}.{operator_index}"
            mutation_log_file = join_path(*mutant_path_parts[:-3], "mutation_logs", f"{bug_name}.log")
            with open(mutation_log_file) as f:
                # append more line hint to final log file
                # line log:  Base.EmailSystem.Util.AOIU_1:32:void_enterElevator(Person):weight += 1 => weight = 1
                mutated_line_hints = f.readlines()[0].split(":", 2)
                mutated_line_hints[0] = bug_name
                joined_line_hint = ":".join(mutated_line_hints)
                final_mutation_log += joined_line_hint

            for feature_name in list_dir(features_dir):
                current_feature_dir = join_path(features_dir, feature_name)
                current_mutated_feature_dir = join_path(current_mutated_features_dir, feature_name)

                if feature_name == package.split(".")[0]:
                    if is_path_exist(current_mutated_feature_dir):
                        # feature dir already copied, keep this and create symlink from mutated java file to this dir
                        if not is_symlink(current_mutated_feature_dir):
                            continue
                        unlink(current_mutated_feature_dir)
                    copy_dir(current_feature_dir, current_mutated_feature_dir)

                else:
                    if not is_path_exist(current_mutated_feature_dir):
                        create_symlink(current_feature_dir, current_mutated_feature_dir)

            mutant_path_dst = join_path(current_mutated_features_dir, *package.split("."), class_name)
            create_symlink(mutant_path, mutant_path_dst)

        # write mutated line position to final mutated log for tracing
        mutated_project_mutation_log_file = join_path(current_mutated_project_dir,
                                                      f"{current_project_name}.mutant.log")
        with open(mutated_project_mutation_log_file, "w+") as f:
            f.write(final_mutation_log)

        mutated_project_dirs.append(current_mutated_project_dir)
    return mutated_project_dirs


TEMP_PROJECT_FOLDER_NAME = ".temp"


def get_mutated_project_dirs(project_dir, include_temp_project_dirs=False, sort=False):
    mutated_projects_dir = get_mutated_projects_dir(project_dir)
    mutated_project_dirs = list_dir(mutated_projects_dir, full_path=True)
    if include_temp_project_dirs:
        temp_mutated_project_dirs = list_dir(join_path(mutated_projects_dir, TEMP_PROJECT_FOLDER_NAME), full_path=True)
        mutated_project_dirs.extend(temp_mutated_project_dirs)
    if sort:
        mutated_project_dirs = natural_sort(mutated_project_dirs)
    return mutated_project_dirs


def get_mutated_project_dir(project_dir, mutated_project_name):
    mutated_projects_dir = get_mutated_projects_dir(project_dir)
    current_mutated_project_dir = join_path(mutated_projects_dir, mutated_project_name)
    if not is_path_exist(current_mutated_project_dir):
        logger.fatal("Can't find mutated project {} from [{}]".format(mutated_project_name, project_dir))
    return current_mutated_project_dir


def get_feature_name_from_mutated_project_name(mutated_project_dir):
    return get_project_name(mutated_project_dir).split(".", 1)[0]


BUG_CONTAINER = {}


def check_bug_from_report(mutated_project_dir, recheck_compilable=False, lib_paths=None):
    configs_report_file_path = get_model_configs_report_path(mutated_project_dir)
    mutated_project_name = get_project_name(mutated_project_dir)
    buggy_statements = sorted(get_multiple_buggy_statements(mutated_project_name, mutated_project_dir))
    key = ";".join(buggy_statements) + ";"
    exist_passing_configuration = False
    exist_failing_configuration = False
    with open(configs_report_file_path, "r") as report_csv:
        reader = csv.reader(report_csv)
        next(reader)
        for i, row in enumerate(reader):
            test_passed = row[-1] == "__PASSED__"
            key += f"{1 if test_passed else 0}"
            if test_passed:
                exist_passing_configuration = True
            else:
                exist_failing_configuration = True
    key += ";"
    is_bug_satisfied = (exist_passing_configuration and exist_failing_configuration)
    if is_bug_satisfied and not BUG_CONTAINER.get(key):
        if recheck_compilable:
            is_compilable = check_all_variant_compilable(mutated_project_dir, lib_paths=lib_paths)
            if not is_compilable:
                return False
        BUG_CONTAINER[key] = True
        return True
    return False