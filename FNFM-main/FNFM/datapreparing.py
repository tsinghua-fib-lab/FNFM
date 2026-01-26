import sys
import os
from typing import Any
import numpy as np
import torch
from numpy import random
from sklearn.datasets import make_moons
import pandas as pd
import random

def datapreparing(targetDataset, basemodel, rawdata_path, kg_emb_path, time_emb_path, graph_set_path, task_params_path):
    if targetDataset=='hill':
        graph_set = np.load(graph_set_path)
        circle=400
        inner=350
        random.seed(42)
        trainid_circle=random.sample(range(inner),int(inner*0.5))
        trainid=[]
        for i in range(9):
            trainid.extend([x + i * circle for x in trainid_circle])
        trainid=list(set(trainid))
        testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
        outerid=range(inner,circle)
    if targetDataset=='HillDisturbe':
        graph_set = np.load(graph_set_path)
        circle=400
        inner=350
        random.seed(42)
        trainid_circle=random.sample(range(inner),int(inner*0.5))
        trainid=[]
        for i in range(3):
            trainid.extend([x + i * circle for x in trainid_circle])
        trainid=list(set(trainid))
        testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
        outerid=range(inner,circle)
    elif targetDataset=='twitter':
        graph_set = np.load(graph_set_path)
        circle=121
        inner=100
        random.seed(42)
        trainid_circle=random.sample(range(inner),int(inner*0.5))
        testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
        trainid=[]
        for i in range(10):
            trainid.extend([x + i * circle for x in trainid_circle])
        trainid=list(set(trainid))
        testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
        outerid=range(inner,circle)
    elif targetDataset=='euroad':
        graph_set = np.load(graph_set_path)
        circle=400
        inner=350
        selected_trajectory=range(len(graph_set))
        random.seed(42)
        trainid_circle=random.sample(range(inner),int(inner*0.5))
        trainid=[]
        for i in range(10):
            trainid.extend([x + i * circle for x in trainid_circle])
        trainid=list(set(trainid))
        testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
        outerid=range(inner,circle)
    elif targetDataset=='collab':
        circle=40
        graph_set=np.load(graph_set_path)
        trainid=[]
        random.seed(42)
        observe_circle=list(range(0,5))+list(range(15,40))
        trainid_circle=list(random.sample(observe_circle,int(len(observe_circle)*0.7)))
        testid=genid=list(random.sample(observe_circle,int(len(observe_circle)*0.3)))
        for i in range(20):
            trainid.extend([x + i * circle for x in trainid_circle])
        outerid=range(5,15)
    elif targetDataset=='fhn2' or targetDataset=='fhn':
        circle=400
        graph_set=np.load(graph_set_path)
        trainid=[]
        e = graph_set[:, 0]
        f = graph_set[:, 1]
        upper_mask = f > e + 0.2
        lower_mask = f < e - 0.2
        upper_ids = np.where(upper_mask)[0]
        lower_ids = np.where(lower_mask)[0]
        observe_circle=list(set[Any](upper_ids) | set(lower_ids))
        random.seed(42)
        trainid_circle=list(random.sample(observe_circle,int(len(observe_circle)*0.7)))
        testid=genid=list(np.setdiff1d(observe_circle,trainid_circle))
        for i in range(10):
            trainid.extend([x + i * circle for x in trainid_circle])
        outerid=np.setdiff1d(range(circle),observe_circle)
    # load full pretrained params
    rawdata = np.load(rawdata_path)
    trainid=np.array(trainid)
    trainid = trainid[trainid < len(rawdata)]
    training_seq = rawdata[trainid]
    genTarget = rawdata[genid]
    channel = 64
    training_seq = training_seq.reshape(training_seq.shape[0],channel,-1)
    genTarget = genTarget.reshape(genTarget.shape[0],channel,-1)
    mean_val = np.mean(training_seq)
    std_val =np.std(training_seq)
    print("means:", mean_val, "stds:", std_val)
    if os.path.exists(kg_emb_path):
        if targetDataset=='Graph':
            kgEmb = np.load(kg_emb_path)
            kgEmb[:, 0]= (kgEmb[:, 0] == 'BA').astype(float)
            kgEmb=kgEmb.astype(np.float32)
        else:
            kgEmb = np.load(kg_emb_path).astype(np.float32)
        
    try:
        if os.path.exists(time_emb_path):
            if targetDataset=='Graph':
                timeEmb = np.load(time_emb_path)
                timeEmb[:, 0]= (timeEmb[:, 0] == 'BA').astype(float)
                timeEmb=timeEmb.astype(np.float32)
            else:
                timeEmb = np.load(time_emb_path).astype(np.float32)
            print(f"success to load timeEmb: {timeEmb.shape}")
    except Exception as e:
        print(f"error: {str(e)}")
        exit(-1)
    kgtrainEmb=kgEmb[trainid].astype(np.float32)
    kggenEmb=kgEmb[genid].astype(np.float32)
    timetrainEmb = timeEmb[trainid].astype(np.float32)
    timegenEmb = timeEmb[genid].astype(np.float32)
    print('timetrainEmb.shape', timetrainEmb.shape)
    print('timegenEmb.shape', timegenEmb.shape)
    return (training_seq.astype(np.float32), 
            (mean_val,std_val), 
            trainid,
            genid,
            outerid, 
            kgtrainEmb,
            kggenEmb, 
            timetrainEmb, 
            timegenEmb, 
            genTarget.astype(np.float32))
