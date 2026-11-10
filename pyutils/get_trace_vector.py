import logging
import os
import time
import pandas as pd

from pyutils.FileManager import list_dir, join_path, get_outer_dir
import xml.etree.ElementTree as ET

def get_spectrum_failed_coverage_inf_new(spectrum_fail_coverage_file, out_exed, out_not_exed):
    exed_data = set()
    not_exed_data = set()
    if os.path.isfile(spectrum_fail_coverage_file):
        data = {}
        #data[variant] = []
        try:
            tree = ET.parse(spectrum_fail_coverage_file)
            root = tree.getroot()
            project = root.find("project")

            for package in project:
                for file in package:
                    file_name = file.get("path")
                    file_name = file_name.split("/")
                    if len(file_name) > 1:
                        file_name = file_name[-1]
                    else:
                        file_name = file_name[0]
                    for line in file:
                        if line.get("num") in out_exed[file_name]:
                            id = line.get('featureClass') + "." + line.get('featureLineNum')
                            exed_data.add(id)
                        elif line.get("num") in out_not_exed[file_name]:
                            id = line.get('featureClass') + "." + line.get('featureLineNum')
                            not_exed_data.add(id)
        except Exception as e:
            logging.info("Exception when parsing %s", spectrum_fail_coverage_file, exc_info=True)
    return exed_data, not_exed_data   # always return a tuple (empty if no failed spectrum -> passing product)

def coverage_filename_to_testname(cf: str) -> str:
    # 只取文件名（防止后面写成路径）
    base = os.path.basename(cf)
    # 去掉 .coverage.xml
    if base.endswith(".coverage.xml"):
        base = base[:-len(".coverage.xml")]
    # ElevatorSystem.Elevator_test00
    dot = base.find(".")
    if dot == -1:
        return base  # 不规范就原样返回
    cls = base[:dot]            # ElevatorSystem
    meth = base[dot + 1:]       # Elevator_test00
    return f"{cls}::{meth}"     # ElevatorSystem::Elevator_test00

def gererateObDataFromSB(production_path, stmt2id):
    oberData = {}

    failed_coverage_path = production_path + "/" + "coverage/failed"
    if os.path.exists(failed_coverage_path):
        failed_coverage_path = join_path(production_path, "coverage/failed")
        current_coverage_file_list = list_dir(failed_coverage_path)  # each coverage file
        spectrum_fail_coverage_file = join_path(get_outer_dir(failed_coverage_path), "spectrum_failed_coverage.xml")

        for cf in current_coverage_file_list:
            out_exed = {}
            out_not_exed = {}
            # #read coverage file
            coverage_file = join_path(failed_coverage_path, cf)
            # if os.path.isfile(coverage_file):
            #     data[variant] = []

            # read
            try:
                tree = ET.parse(coverage_file)
                root = tree.getroot()

                project = root.find("project")

                start = False
                for package in project:
                    for file in package:
                        file_name = file.get("name")
                        if file_name != None:
                            out_exed[file_name] = set()
                            out_not_exed[file_name] = set()
                            for line in file:
                                if line.tag == "line":
                                    if line.attrib.get("truecount") != None:
                                        num = line.attrib.get("num")
                                        if line.attrib.get("truecount") == "0":
                                            if num in out_exed[file_name]:
                                                out_exed[file_name].remove(num)
                                            out_not_exed[file_name].add(num)
                                        else:
                                            if num in out_not_exed[file_name]:
                                                out_not_exed[file_name].remove(num)
                                            out_exed[file_name].add(num)
                                    elif line.attrib.get("num") != None:
                                        num = line.attrib.get("num")
                                        if line.attrib.get("count") == "0":
                                            out_not_exed[file_name].add(num)
                                        else:
                                            out_exed[file_name].add(num)
            except:
                logging.info("Exception when parsing %s", coverage_file, exc_info=True)

            exed_data, not_exed_data = get_spectrum_failed_coverage_inf_new(spectrum_fail_coverage_file, out_exed, out_not_exed)


            temp = []
            for stmt in stmt2id:
                if stmt in exed_data:
                    temp.append(1)
                else:
                    temp.append(0)
            test_name = coverage_filename_to_testname(cf)
            oberData[test_name] = temp
    return oberData


# nodePath = "/home/whn/codes/Static_Slicing-master/Static_Slicing-master/output/nodes.txt"


def get_spectrum_stm(spectrum_fail_coverage_file, stmt2id):
    if os.path.isfile(spectrum_fail_coverage_file):
        try:
            tree = ET.parse(spectrum_fail_coverage_file)
            root = tree.getroot()
            project = root.find("project")

            for package in project:
                for file in package:
                    for line in file:
                        id = line.get('featureClass') + "." + line.get('featureLineNum')
                        if id not in stmt2id:
                            cur_id = len(stmt2id) + 1
                            stmt2id[id] = cur_id

        except Exception as e:
            logging.info("Exception when parsing %s", spectrum_fail_coverage_file, exc_info=True)
    return stmt2id   # always return stmt2id (unchanged if no failed spectrum -> passing product)
