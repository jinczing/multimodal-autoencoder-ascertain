import subprocess
import os, sys, traceback, getpass, time, re
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def libsvm_to_mmae_csv(output_path, svm_path, modality_names, modality_nums):
    """
    args:
        output_path: a path to place csv file
        svm_path: source libsvm style file path (0:subject_id, 1:clip_id)
    
    return:
    csv file with column: arousal_label, valence_label, subject_id, clip_id, dataset, logistics_noisy, [modality_feature_id]
    """
    samples = open(svm_path).read().split('\n')
    
    f = open(output_path, 'w+')
    
    # write column to csv
    f.write('index' + ',' + 'arousal_label' + ',' + 'valence_label' + ',' + 'subject_id' + ',' + 'clip_id' + ',' + 'dataset' + ',' + 'logistics_noisy')
    for modality_index in range(len(modality_names)):
        for feature_index in range(modality_nums[modality_index]):
            f.write(',' + modality_names[modality_index] + '_' + str(feature_index+1))
    
    injected_index = [] # [4]
    injected_column = [] # [False]
    
    idd = 0
    for sample in samples:
        index = 0
        if sample == '': continue
        f.write('\n')
        for txt in sample.split():
            if index in injected_index:
                f.write(',')
                f.write(str(injected_column[injected_index.index(index)]))
            if txt=='1' or txt=='0':
                f.write(str(idd+1) + ',')
                f.write(txt)
            else:
                f.write(',')
                f.write(txt.split(':')[1])
            index += 1
        idd += 1
                
    f.close()
        
        
    

def get_alpha(distances):
    return [(1/distances[0])/(1/distances[0]+1/distances[1]+1/distances[2]), (1/distances[1])/(1/distances[0]+1/distances[1]+1/distances[2]), 
           (1/distances[2])/(1/distances[0]+1/distances[1]+1/distances[2])]

def fusion_predict(resource_path, dataset_paths, names, f1_scores, alphas, optimized_alphas, k, cs, gs):
    output_names = [name + '_output' for name in names]
    train_d = [[], [], []]
    
    for i in range(len(dataset_paths)):
        dataset = open(dataset_paths[i], 'r').read().split('\n')
        batch = len(dataset)//k
        for j in range(k):
            if j != k-1: # not last round
                train_d[i].append(dataset[j*batch:(j+1)*batch])
            else:
                train_d[i].append(dataset[j*batch:])
    
    f1s = []
    aucs = []
    for v in range(k):
        predicts = [[], [], []]
        trues = []
        results = []
        for i in range(len(dataset_paths)):
            train_name = names[i] + '_train_' + str(v+1) + '_' + str(i+1)
            test_name = names[i] + '_test_' + str(v+1) + '_' + str(i+1)
            train_path = './tmp/' + train_name
            test_path = './tmp/' + test_name

            f = open(train_path, 'w+') # create a new train dataset
            for j in range(k):
                if j != v:
                    jindex = 0
                    for txt in train_d[i][j]:
                        f.write(txt)
                        if jindex != len(train_d[i][j])-1 or j != k-1:
                            f.write('\n')
                        jindex += 1
            f.close()

            f = open(test_path, 'w+') # create a new test dataset
            for j in range(k):
                if j == v:
                    jindex = 0
                    for txt in train_d[i][j]:
                        f.write(txt)
                        if jindex != len(train_d[i][j])-1:
                            f.write('\n')
                        jindex += 1
            f.close()
            
            model_path = './' + names[i] + '_train_' + str(v+1) + '_' + str(i+1) + '.model'
            output_path = './' + output_names[i] + '_' + str(v+1) + '_' + str(i+1)
            
            cmdline = get_train_cmd(resource_path+'svm-train.exe', train_path, cs[i], gs[i], 1)
            process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
            process.wait()

            cmdline = get_test_cmd(resource_path + 'svm-predict.exe', test_path, model_path, output_path)
            process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
            process.wait()

            output_file = open(output_path).read().split('\n')[1:-1]
            for txt in output_file:
                predicts[i].append(float(txt.split()[1]))
            if i == 0:
                true_file = open(test_path).read().split('\n')
                for txt in true_file:
                    if txt == '': continue
                    trues.append(int(txt.split()[0]))
        ws = []
        alpha_total = sum([alphas[x]*f1_scores[x] for x in range(len(dataset_paths))])
        for i in range(len(dataset_paths)):
            ws.append(optimized_alphas[i] * f1_scores[i] * alphas[i] / alpha_total)
        
        for i in range(len(predicts[0])):
            weighted_score = sum([ws[x] * predicts[x][i] for x in range(len(dataset_paths))])
            results.append(weighted_score)
        
        f1 = f1_score(results, trues)
        fpr, tpr, _ = roc_curve(trues, results)
        aucauc = auc(fpr, tpr)
        print(f1, aucauc)
        f1s.append(f1)
        aucs.append(aucauc)
    return f1s, aucs
        

def weighted_predict(identification, personality, resource_path, dataset_path, f1_scores=[0.819, 0.815, 0.762], centroids=[[3.917, 5.413, 5.322],[3.744, 4.67, 3.609],[5.666, 5.482, 3.51]]):
    personality = np.array(personality)
    centroids = np.array(centroids)
    distances = []
    correct_num = 0
    dataset = open(dataset_path, 'r').read().split('\n')
    output_names = [identification + '_' + str(x) + '_output' for x in range(3)]
    output_files = [[],[],[]]
    print(identification, dataset_path)
    
    for i in range(3): # calculate distance
        distances.append(np.linalg.norm(personality-centroids[i]))
    
    for i in range(3): # predict by all svms
        model_name = 'arousal_train_' + str(i+1) + '_cluster.model'
        output_name = output_names[i]
        cmdline = get_test_cmd(resource_path + 'svm-predict.exe', dataset_path, resource_path +  model_name, output_name)
        process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        process.wait()
        output_files[i].append(open('./' + output_name).read().split('\n')[1:-1])
    
    f = open('./' + identification + '_output', 'w+')
    
    for i in range(len(output_files[0][0])):
        x = float(output_files[0][0][i].split()[1])
        y = float(output_files[1][0][i].split()[1])
        z = float(output_files[2][0][i].split()[1])
        alphas = get_alpha(distances)
        f1_divisor = sum([alphas[x]*f1_scores[x] for x in range(3)])
        tx = f1_scores[0] / f1_divisor
        ty = f1_scores[1] / f1_divisor
        tz = f1_scores[2] / f1_divisor
        score = alphas[0]*x*tx + alphas[1]*y*ty + alphas[2]*z*tz
        f.write(str(score) + '\n')
        for s in dataset[i].split():
            if s[0] == '+':
                if score > 0.5:
                    correct_num += 1
            else:
                if score <= 0.5:
                    correct_num += 1
            break

    
    f.close()
    
    return correct_num / len(output_files[0][0])

def performance(dataset_file, output_file):
    dataset_file = dataset_file.split('\n')
    
    predicts = []
    trues = []
    
    # predicts 
    output_file = output_file.split('\n')[1:-1]
    for x in output_file:
        if x == '': break
        xx = x.split()
        predicts.append(float(xx[1]))
    tp_num = 0
    correct_num = 0
    true_num = 0
    
    xx = [1 if x>0.5 else 0 for x in predicts]
    total = sum(xx)
    
    # trues
    index = 0
    for d in dataset_file:
        for s in d.split():
            if s[0] == '+':
                trues.append(1)
                true_num += 1
                if predicts[index] > 0.5:
                    tp_num += 1
                    correct_num += 1
            else:
                trues.append(-1)
                if predicts[index] <= 0.5:
                    correct_num += 0
            break
        index += 1
    
    
    #total = len(trues)
    precision = 0
    if total == 0:
        precision = 0
    else:
        precision = correct_num/total

    if true_num == 0:
        recall = 0
    else:
        recall = tp_num / true_num
    f1_score = 0
    if (precision+recall) != 0:
        f1_score = 2 * (precision*recall)/(precision+recall)
    else:
        f1_score = 0
    return predicts, trues, f1_score, precision

def predict():
    pass

def f1_score(predicts, trues):
    positive_count = 0
    tp = 0
    true_count = 0
    for i in range(len(predicts)):
        if predicts[i] > 0.5:
            positive_count += 1
            if trues[i] == 1:
                tp += 1
                true_count += 1
        else:
            if trues[i] == 1:
                true_count += 1
    precision = 0
    if positive_count == 0:
        precision = 0
    else:
        precision = tp / positive_count
        
    recall = 0
    if true_count == 0:
        recall = 0
    else:
        recall = tp / true_count
    
    f1 = 2 * (precision*recall)/(precision + recall) if (precision + recall) != 0 else 0
    
    return f1

def independent_test(resource_path, dataset_path, train_name, test_name, c=4, g=0.0078125, b=1):
    # train model
    cmdline = get_train_cmd(resource_path + 'svm-train.exe', dataset_path+train_name, c, g, b)
    process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
    process.wait()
    result = process.stdout

    # predict model
    cmdline = get_test_cmd(resource_path + 'svm-predict.exe', dataset_path+test_name, train_name + '.model', test_name + '_output')
    process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
    process.wait()
    result = process.stdout

    dataset_file = open(dataset_path+test_name).read()
    output_file = open('./' + test_name + '_output').read()

    predicts, trues, f1, precision = performance(dataset_file, output_file)
    return predicts, trues
        
    
def f1_score_evaluate(resource_path, dataset_path, dataset_name, k=10, c=4, g=0.0078125, b=1):
    f = open(dataset_path, 'r')
    data = f.read()
    f.close()
    data = data.split('\n')
    batch = len(data)//k
    train_d = []
    # slice data
    for i in range(k):
        if i != k-1: # not last round
            train_d.append(data[i*batch:(i+1)*batch])
        else:
            train_d.append(data[i*batch:])
            
    
            
    f1s = []
    aucs = []
    accs = []
    
    predss = []
    truess = []


    for i in range(k): # k folds validation
        train = dataset_name + '_train_' + str(i+1)
        test = dataset_name + '_test_' + str(i+1)
        train_name = './tmp/' + train
        test_name = './tmp/' + test
        
        f = open(train_name, 'w+') # create a new train dataset
        for j in range(k):
            if j != i:
                jindex = 0
                for txt in train_d[j]:
                    f.write(txt)
                    if jindex != len(train_d[j])-1 or j != k-1:
                        f.write('\n')
                    jindex += 1
        f.close()
        
        f = open(test_name, 'w+') # create a new test dataset
        for j in range(k):
            if j == i:
                jindex = 0
                for txt in train_d[j]:
                    f.write(txt)
                    if jindex != len(train_d[j])-1:
                        f.write('\n')
                    jindex += 1
        f.close()
        
        # train model
        cmdline = get_train_cmd(resource_path + 'svm-train.exe', train_name, c, g, b)
        process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        process.wait()
        result = process.stdout
        
        # predict model
        cmdline = get_test_cmd(resource_path + 'svm-predict.exe', test_name, train + '.model', test + '_output')
        process = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        process.wait()
        result = process.stdout
        
        dataset_file = open(test_name).read()
        output_file = open('./' + test + '_output').read()
        
        predicts, trues, f1, precision = performance(dataset_file, output_file)
        predss.append(predicts[0])
        truess.append(trues[0])
        
        #fpr, tpr, _ = roc_curve(trues, predicts)
        #aucauc = auc(fpr, tpr)
        

        
        f1s.append(f1)
        #aucs.append(aucauc)
        accs.append(precision)
        
        #print(f1, aucauc)
    return predss, truess
        
        
        
def get_train_cmd(exe_path, dataset_path, c, g, b):
    #print(('"' + exe_path + '"' + ' -c ' + str(c) + ' -g ' + str(g) + ' -b ' + str(b) + ' "' + dataset_path + '"'))
    return ('"' + exe_path + '"' + ' -c ' + str(c) + ' -g ' + str(g) + ' -b ' + str(b) + ' "' + dataset_path + '"')

def get_test_cmd(exe_path, dataset_path, model_path, output_path):
    #print('"' + exe_path + '" ' + '"' + dataset_path + '" ' + '"' + model_path + '" ' + '"' + output_path + '"')
    return ('"' + exe_path + '" ' + '-b 1 ' + '"' + dataset_path + '" ' + '"' + model_path + '" ' + '"' + output_path + '"')
        