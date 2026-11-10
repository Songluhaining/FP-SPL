import os
from re import template
import sys
import pylib.spl as fl
import time
import func_timeout
import random
import numpy as np


def runtesttrace(cmdarg):
    bindir = os.path.abspath("./target")
    cmdargs = cmdarg.split('#')
    cmdstr = " mvn jar:jar && mvn test -Dtest={classname}#{testname} -DargLine=\"-noverify -javaagent:{jardir}/ppfl-0.0.1-SNAPSHOT.jar=logfile={classname}.{testname},instrumentingclass=trace.{classname}\"".format(
        classname=cmdargs[0], testname=cmdargs[1], jardir=bindir)
    print(cmdstr)
    os.system(cmdstr)


def runtest(cmdarg):
    cmdstr = "mvn test -Dtest={testname}".format(testname=cmdarg)
    print(cmdstr)
    os.system(cmdstr)


def makedirs():
    dirs2make = ["./configs", "./test/trace/patterns", "./test/trace/logs",
                 "./trace/runtimelog/", "./test/trace/logs/mytrace", "./d4j_resources/metadata_cached/"]
    for dir in dirs2make:
        if not(os.path.exists(dir)):
            os.makedirs(dir)

def modify(decay,site):
    changes = 0.5*(0.9**(decay))#0.25 0.5
    data = np.loadtxt('infer.txt',delimiter=',')
    f = open('infer.txt', 'w+', encoding='utf-8', errors='ignore')
    s = ''
    a = []

    ori = data[site]
    while ori == data[site]:
        data[site] = data[site] + changes*(random.random()*2-1)
        if data[site] > 1:
            data[site] = 1.0
        if data[site] < 0:
            data[site] = 0.0

    a = data
    for i in range(len(a)):
        if i<len(a)-1:
            s = s + str(a[i])+','
        else:
            s = s + str(a[i])
    f.write(s)
    f.close()

def modifyback(data):
    f = open('infer.txt', 'w+', encoding='utf-8', errors='ignore')
    s = ''
    for i in range (len(data)):
        if i<len(data)-1:
            s = s + str(data[i])+','
        else:
            s = s + str(data[i])
    f.write(s)
    f.close()

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments')

    # Argument to run one specific experiment
    parser.add_argument('--system', help='System name', default="BankAccountTP")
    # Argument to run one specific SPL system
    parser.add_argument('--buggy_systems_folder', help='Run specific SPL',
                        default="./examples/4wise-BankAccountTP-1BUG-Full")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    fl.fl(False)