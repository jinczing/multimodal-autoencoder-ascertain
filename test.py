import subprocess
import os, sys, traceback, getpass, time, re

dirname = ''
exe_path = os.path.join(dirname, r'./libsvm-3.23/windows/svm-train.exe')
dataset_path = os.path.join(dirname, r'./svmlib_dataset/arousal_train_3_cluster')
print(exe_path)
if not os.path.exists(exe_path):
    raise IOError('svm-train executable not found')
if not os.path.exists(dataset_path):
    raise IOError('dataset not found')
cmdline = '"' + exe_path  + '"' + ' -c 4 -g 0.0078125 -b 1 ' + '"' + dataset_path + '"'
cmdline = '"./libsvm-3.23/windows/svm-predict.exe" "./tmp/arousal_train_1_cluster_test_2" "arousal_train_1_cluster_train_2.model" "arousal_train_1_cluster_test_2_output"'
result = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
a = result.stdout
for line in a.readlines():
    print(str(line))
#     if str(line).find('Cross') != -1:
#         return float(line.split()[-1][0:-1])