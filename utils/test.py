from subprocess import *
import os, sys, traceback, getpass, time, re

dirname = os.path.dirname(__file__)
exe_path = os.path.join(dirname, "./libsvm-3.23/windows/svm-train.exe")
dataset_path = os.path.join(dirname, "./svmlib_dataset/arousal_train_3_cluster")
cmdline = exe_path + " -c 4 -g 0.0078125 -b 1 " + dataset_path
result, error = Pop(cmdline,shell=True,stdout=PIPE,stderr=PIPE,stdin=PIPE).stdout
print(error)
print(result)
for line in result.readlines():
    print(str(line))