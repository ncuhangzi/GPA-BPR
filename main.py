# coding=utf-8
from GPBPR import GPBPR
from GPABPR import GPABPR
import torch

import os
from torch.utils.data import TensorDataset,DataLoader
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_
import numpy as np
import math
from torch.optim import Adam
from sys import argv
from RatingDataset import TrainDataset, TestDataset
from GPUUtil import set_device, move_to_device, move_model_to_device

"""
    my_config is a dict which contains necessary filepath in trainning and evaluating GP-BPR model

    visual_features is the output of last avgpool in resnet50 of torchvision, obtained by 

    textural_features is the input of word embeding layer

    embedding_matrix is the word embedding vector from nwjc2vec. Missing word initialed as zero vector 
"""

my_config = {
    "visual_features_dict": "./feat/visualfeatures",
    "textural_idx_dict": "./feat/textfeatures",
    "textural_embedding_matrix": "./feat/smallnwjc2vec",
    "train_data": r"./data/train.csv",
    "valid_data": r"./data/valid.csv",
    "test_data": r"./data/test.csv",
    "model_file": r"./model/GPABPR_V.model",
}
root = os.path.dirname(os.path.realpath(__file__)) + "/data/IQON3000/"
metapath_files = ["pmf_outfitmetapath_005.txt"]
#feature 可能也不用匯入 GPBPR本身有替user item生成embedding可是維度可能和MCRec要得不一樣
feature_file_dict = {"u": "user_node_emb.txt","t": "top_node_emb.txt", "b": "bottom_node_emb.txt"}

train_data_file = "user_bottom_train.txt"
# test_data_file = "user_bottom_test.txt"

for i in range(len(metapath_files)):
    metapath_files[i] = root + metapath_files[i]

for feature_type, feature_file_name in feature_file_dict.items():
    feature_file_dict[feature_type] = root + feature_file_name

# #要改
# train_dataset = TrainDataset(root + train_data_file, metapath_files, negative_num, feature_file_dict)
# test_dataset = TestDataset(train_dataset, root + test_data_file)

# train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
# test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

max_user_id, max_item_id = 0, 0

with open(root + train_data_file, "r") as input:
    for line in input.read().splitlines():
        arr = line.split(" ")
        u, i = int(arr[0]), int(arr[1])
        max_user_id = max(max_user_id, u)
        max_item_id = max(max_item_id, i)

#load metapath data成特定輸入格式 => list of metapaths
metapath_list = []

def load_metapath(metapath_files):

    type2id = {}
    id2type = {}
    tmp_type_set = set()

    for metapath_file in metapath_files:
        with open(metapath_file) as input:
            for line in input.read().splitlines():
                arr = line.split(' ')
                
                for nodes in arr[0].split('-'):
                    tmp_type_set.add(nodes[0])

    for node_type in tmp_type_set:
        id = len(id2type)
        type2id[node_type] = id
        id2type[id] = node_type

    print("node type " + str(type2id))

    

    for metapath_file in metapath_files:
        path_dict = {}
        max_path_num = 0
        hop_num = 0

        with open(metapath_file) as input:
            for line in input.read().splitlines():
                arr = line.split(' ')
                max_path_num = 1
                hop_num = len(arr[0].split('-'))
        #create the path_dict
        with open(metapath_file) as input:
            for line in input.read().splitlines():
                arr = line.strip().split(' ')
                path = arr[0]
                tmp = path.split('-')
                for node in tmp:
                    index = node[1:]
                    if node[0] == 'u':
                        u = int(index)
                    elif node[0] == 'b':
                        i = int(index)

                path_dict[(u, i)] = []


        with open(metapath_file) as input:
            for line in input.read().splitlines():
                arr = line.strip().split(' ')
                # u, i = arr[0].split(',')
                # u, i = int(u), int(i)
                # u,i = 0,0

                # path_dict[(u, i)] = []

                path = arr[0]
                # tmp = path.split(' ')[0].split('-')
                tmp = path.split('-')
                node_list = []
                for node in tmp:
                    #index = int(node[1:]) - 1 #why -1?
                    index = node[1:]
                    if node[0] == 'u':
                        u = int(index)
                    elif node[0] == 'b':
                        i = int(index)

                    node_list.append([type2id[node[0]], index]) #檢索條件不一樣 需要調整
                # path_dict[(u, i)] = (node_list)
                path_dict[(u, i)].append(node_list)
                max_path_num = max(len(path_dict[(u, i)]), max_path_num)

        metapath_list.append((metapath_file, path_dict, max_path_num, hop_num, max_user_id, max_item_id,id2type))

load_metapath(metapath_files)
metapath_list_attributes = []
#metapath_list_attributes includes path_num, hop_num
# metapath_list[i]: (metapath_file, path_dict, path_num, hop_num,max_user_id, max_item_id) 
for i in range(len(metapath_list)):
    metapath_list_attributes.append((metapath_list[i][2], metapath_list[i][3]))# 2,3


def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result
def load_embedding_weight(device):
    jap2vec = torch.load(my_config['textural_embedding_matrix'])
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight


def trainning(model, mode, train_data_loader, device, visual_features, text_features, opt):
    r"""
        using data from Args to train model

        Args:

            mode: -

            train_data_loader: mini-batch iteration

            device: device on which model train

            visual_features: look up table for item visual features

            text_features: look up table for item textural features

            opt: optimizer of model
    """
    model.train()
    model = model.to(device)
    for iteration,aBatch in enumerate(train_data_loader):
        output , outputweight = model.fit(aBatch[0], visual_features, text_features, weight=False)
         # print(output.size())
        loss = (-logsigmoid(output)).sum() + 0.001*outputweight
        iteration += 1
        opt.zero_grad()
        loss.backward()
        opt.step()

def evaluating(model, mode, test_csv, visual_features, text_features):
    r"""
        using data from Args to train model

        Args:

            mode: -

            train_data_loader: mini-batch iteration

            test_csv: valid file or test file

            visual_features: look up table for item visual features

            text_features: look up table for item textural features
    """
    model.eval()
    testData = load_csv_data(test_csv)
    pos = 0
    MRR =0
    batch_s = 100 #所以output 為100維之vector
    key = 0
    for i in range(0, len(testData), batch_s): #start, stop, stride

        data = testData[i:i+batch_s] if i+batch_s <=len(testData) else testData[i:]
        output, candidates = model.forward(data, visual_features, text_features, 70)
        if(key == 0):
            #print(output)  #100維的tensor
            #print('sum of output.ge(0) : \n')
            #print(torch.sum(output.ge(0)))
            sort_cand = []
            targets =[]
            for item in candidates:
                sort_cand.append(sorted(item, reverse=True))
                targets.append(item[0])

            indicies = torch.LongTensor(sort_cand).cuda()
            targets = torch.LongTensor(targets).cuda()
            mrr = get_mrr(indicies, targets)
            #key = 1
            #print(mrr, "  /")

        MRR += mrr
        pos += float(torch.sum(output.ge(0))) #用ge function將output值跟0相比，較0大為true較0小為false，將output元素相加 有包含True的output就會加1
    print( "evaling process: " , test_csv , model.epoch, pos/len(testData), "MRR: ", MRR/math.ceil((len(testData)/batch_s))) #剩下資料不到batch size即小於100，會使分母變小，MRR而因此變大
    #print( "Mrr : ", rr/len(testData))

def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model. model  predict出來的前k個index
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    tmp_rank = []
    np_hits = hits.cpu().detach().numpy()
    for i in range(np_hits.shape[0]):
        if i >=1:
            if np_hits[i][0] == np_hits[i-1][0]:
                continue
        tmp_rank.append(np_hits.tolist()[i])
    tmp_rank = torch.LongTensor(tmp_rank).cuda()
    #只取順位高的一個TARGET
    #ranks = hits[:, -1] + 1 #索引轉換為順位 索引從0開始 順位從1開始 所以要加一
    ranks = tmp_rank[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()

def F(mode ,hidden_dim, batch_size, device):
    print('loading top&bottom features')
    # torch.cuda.set_device("")
    train_data = load_csv_data(my_config['train_data'])

    visual_features = torch.load(my_config['visual_features_dict'], map_location= lambda a,b:a.cpu())
    
    text_features = torch.load(my_config['textural_idx_dict'], map_location= lambda a,b:a.cpu())

    try:
        print("loading model")
        gpbpr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(device))
    except Exception as e:
        print(e)
        print('no module exists, created new one {}'.format(my_config['model_file']))
        embedding_weight = load_embedding_weight(device)
        item_set= set()
        user_set = set([str(i[0]) for i in train_data])
        for i in train_data:
            item_set.add(str(int(i[2])))
            item_set.add(str(int(i[3])))
        # move_to_device(metapath_list)
        # move_to_device(metapath_list_attributes)
        gpbpr = GPABPR(user_set = user_set, item_set = item_set, 
            embedding_weight=embedding_weight,metapath_feature=metapath_list,metapath_list_attributes=metapath_list_attributes, uniform_value = 0.3).to(device)
    
    opt = Adam([
    {
        'params': gpbpr.parameters(),
        'lr': 0.0005,
    }
    ])

    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.int))
    train_loader = DataLoader(train_data, batch_size= batch_size,shuffle=True, drop_last=True)


    for i in range(40):

        # 这里是单个进程的训练
        trainning(gpbpr, mode, train_loader,device, visual_features, text_features, opt)
        
        
        gpbpr.epoch+=1
        torch.save(gpbpr, my_config['model_file'])

        evaluating(gpbpr,mode, my_config['valid_data'],  visual_features, text_features,)
        evaluating(gpbpr,mode, my_config['test_data'],  visual_features, text_features,)


if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)
    import os
    try:
        os.mkdir('./model')
    except Exception: pass
    F(mode = 'final', hidden_dim = 512, batch_size = 256, device = 0)