from zlib import DEFLATED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np


class UIEmbedding(nn.Module):

    def __init__(self, latent_dim, obj_num):
        super(UIEmbedding, self).__init__()

        self.latent_dim = latent_dim
        # id starts from 1, add one more id 0 for invalid updates
        self.embedding = nn.Embedding(num_embeddings=obj_num + 1, embedding_dim=latent_dim)

        nn.init.xavier_normal_(self.embedding.weight.data)

    def forward(self, input):
        # input.shape: batch_size, negative_num + 1, latent_dim
        input = torch.from_numpy(input).cuda()
        input = self.embedding(input)
        input = input.view(-1, self.latent_dim)

        return input


class MetaPathEmbedding(nn.Module):

    def __init__(self, path_num, hop_num, feature_size, latent_dim):
        super(MetaPathEmbedding, self).__init__()

        self.path_num = path_num
        self.hop_num = hop_num
        self.feature_size = feature_size
        self.latent_dim = latent_dim
        # self.lam = lambda x, index: x[:, index, :, :]
        global Lam
        def Lam(x, index):
            return x[:, index, :, :]
        self.lam = Lam

        if hop_num == 3:
            kernel_size = 3
        elif hop_num == 4:
            kernel_size = 4
        else:
            print(hop_num)
            raise Exception("Only support 3-hop or 4-hop metapaths, hop %d" % (hop_num))

        # channel: number of dimensions of the embeddings
        self.conv1D = nn.Conv1d(in_channels=self.feature_size, out_channels=self.latent_dim, kernel_size=kernel_size,
                                stride=1,
                                padding=0)
        nn.init.xavier_uniform_(self.conv1D.weight.data)

        # ToDo: Not necessary???
        # self.gMaxPooling = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        
        input = torch.from_numpy(np.asarray(input)).cuda()
        # input.shape: batch_size, negative_num + 1, path_num, hop_num， feature_size

        input = input.view(-1, self.path_num, self.hop_num, self.feature_size)
        # print('input size:', input.size())
        # input.shape: batch_size * (negative_num + 1), path_num, hop_num, feature_size

        # Step 1 Path Instance Embedding: concatenate embeddings of nodes on the metapath
        path_input = self.lam(input, 0)
        # print('path_input',path_input.size())
        # path_input.shape: batch_size * (negative_num + 1), hop_num, feature_size

        # Conv1d expects (batch, in_channels, in_length). 維度換位
        path_input = path_input.permute(0, 2, 1)
        global output
        output = self.conv1D(path_input).permute(0, 2, 1)
        output = F.relu(output)

        # output = self.gMaxPooling(output)

        output = self.dropout(output)
        #lam:抽出某index上的path
        for i in range(1, self.path_num):
            path_input = self.lam(input, i)
            path_input = path_input.permute(0, 2, 1)
            tmp_output = self.conv1D(path_input).permute(0, 2, 1)
            tmp_output = F.relu(tmp_output)

            # tmp_output = self.gMaxPooling(tmp_output)

            tmp_output = self.dropout(tmp_output)
            output = torch.cat((output, tmp_output), 2)

        output = output.view((-1, self.path_num, self.latent_dim))
        # print('output size:', output.size())
        # Step 2 Metapath embedding
        output = torch.max(output, 1, keepdim=True)[0]
        # batch_size * (negative_num + 1), 1, latent_dim
        # print('output size(max):', output.size())
        
        return output


class UIAttention(nn.Module):

    def __init__(self, latent_dim, att_size):
        super(UIAttention, self).__init__()

        self.dense = nn.Linear(in_features=latent_dim * 2, out_features=att_size)
        nn.init.xavier_normal_(self.dense.weight.data)
        # global Lam4
        # def Lam4(x):
        #     return F.softmax(x, dim=1)
        # self.lam = lambda x: F.softmax(x, dim=1)
        self.lam4 = Lam2

    def forward(self, input, path_output):
        inputs = torch.cat((input, path_output), 1)

        output = self.dense(inputs)
        output = torch.relu(output)

        atten = self.lam4(output)

        # element-wise produt
        output = input * atten

        return output


class MetaPathAttention(nn.Module):

    def __init__(self, att_size, latent_dim, metapath_type_num):
        super(MetaPathAttention, self).__init__()

        self.att_size = att_size
        self.latent_dim = latent_dim
        self.metapath_type_num = metapath_type_num #len of meta_attribute_list

        self.dense_layer_1 = nn.Linear(in_features=latent_dim * 3, out_features=att_size)
        self.dense_layer_2 = nn.Linear(in_features=att_size, out_features=1)
        nn.init.xavier_normal_(self.dense_layer_1.weight.data)
        nn.init.xavier_normal_(self.dense_layer_2.weight.data)

        # self.lam1 = lambda x, index: x[:, index, :]
        global Lam1, Lam2, Lam3
        def Lam1(x, index):
            return x[:, index, :]
        def Lam2(x):
            return  F.softmax(x, dim=1)
        def Lam3(metapath_latent, atten):
            return torch.sum(metapath_latent * torch.unsqueeze(atten, -1), 1)
        self.lam1 = Lam1
        self.lam2 = Lam2
        self.lam3 = Lam3

        # self.lam1 = lambda x, index: x[:, index, :]
        # self.lam2 = lambda x: F.softmax(x, dim=1)
        # self.lam3 = lambda metapath_latent, atten: torch.sum(metapath_latent * torch.unsqueeze(atten, -1), 1)

    def forward(self, user_latent, item_latent, metapath_latent):
        # metapath_latent: batch_size * negative_num_plus, metapath_type_num, latent_dim
        metapath = self.lam1(metapath_latent, 0)
        # print('user latent', user_latent.size())
        # print('item latent', item_latent.size())
        # print('metapath latent', metapath.size())
        inputs = torch.cat((user_latent, item_latent, metapath), 1)

        output = self.dense_layer_1(inputs)
        output = F.relu(output)

        output = self.dense_layer_2(output)
        output = F.relu(output)

        for i in range(1, self.metapath_type_num):
            metapath = self.lam1(metapath_latent, i)
            inputs = torch.cat((user_latent, item_latent, metapath), 1)

            tmp_output = self.dense_layer_1(inputs)
            tmp_output = F.relu(tmp_output)

            tmp_output = self.dense_layer_2(tmp_output)
            tmp_output = F.relu(tmp_output)

            output = torch.cat((output, tmp_output), 1)

        atten = self.lam2(output)

        output = self.lam3(metapath_latent, atten)

        return output


class MCRec(nn.Module):

    def __init__(self, latent_dim, att_size, feature_size, negative_num, user_num, item_num, metapath_list_attributes,
                 layer_size):
        super(MCRec, self).__init__()

        self.latent_dim = latent_dim
        self.att_size = att_size
        self.feature_size = feature_size
        self.negative_num = negative_num
        self.user_num = user_num
        self.item_num = item_num

        self.user_latent = UIEmbedding(latent_dim, user_num)
        self.item_latent = UIEmbedding(latent_dim, item_num+100000)

        self.path_latent_vecs = nn.ModuleList()
        global MetaPathEmbedding
        # metapath_list_attributes[i]: (path_num, hop_num)
        for i in range(len(metapath_list_attributes)):
            # print('path:',metapath_list_attributes[i][0])
            # print('hop:',metapath_list_attributes[i][1])
            # print('feature_size:',self.feature_size)
            # print('latent dim:', self.latent_dim)
            metapath_emb = MetaPathEmbedding(path_num=metapath_list_attributes[i][0],
                                             hop_num=metapath_list_attributes[i][1], feature_size=self.feature_size,
                                             latent_dim=self.latent_dim)
            self.path_latent_vecs.append(metapath_emb)

        self.metapath_att = MetaPathAttention(att_size=self.att_size, latent_dim=self.latent_dim,
                                              metapath_type_num=len(metapath_list_attributes))
        self.user_att = UIAttention(latent_dim=self.latent_dim, att_size=self.att_size)
        self.item_att = UIAttention(latent_dim=self.latent_dim, att_size=self.att_size)

        self.layers = nn.ModuleList()

        assert len(layer_size) > 0

        dense_layer = nn.Linear(in_features=self.att_size * 3, out_features=layer_size[0])
        self.layers.append(dense_layer)

        for i in range(1, len(layer_size)):
            dense_layer = nn.Linear(in_features=layer_size[i - 1], out_features=layer_size[i])
            self.layers.append(dense_layer)

        self.predict = nn.Linear(in_features=layer_size[-1], out_features=1)

    def forward(self, user_input, item_input, metapath_inputs):

        # user_input.shape/item_input.shape: batch_size, negative_num + 1
        # metapath_inputs: num_of_metapath_types, batch_size, negative_num + 1, path_num, latent_dim, hop_num
        path_output = None
        for i in range(len(metapath_inputs)):
            
            output = self.path_latent_vecs[i](tuple(metapath_inputs[i]))

            if path_output is None:
                path_output = output
            else:

                path_output = torch.cat((path_output, output), 2)
        # print('before path output',path_output.size())
        # batch_size * negative_num_plus, latent_dim, metapath_type
        # path_output = path_output.view((len(metapath_inputs), -1, self.latent_dim))
        path_output = path_output.view((-1,len(metapath_inputs), self.latent_dim))
        # print('after path output',path_output.size())
        user_input = self.user_latent(user_input)
        item_input = self.item_latent(item_input)

        path_atten = self.metapath_att(user_input, item_input, path_output)
        #set checkpoint
        # user_atten = checkpoint(self.user_att(user_input, path_atten))
        # item_atten = checkpoint(self.item_att(item_input, path_atten))
        user_atten = self.user_att(user_input, path_atten)
        item_atten = self.item_att(item_input, path_atten)

        output = torch.cat((user_atten, path_atten, item_atten), 1)
        # print(output)
        for layer in self.layers:
            output = layer(output)
            output = F.relu(output)
        
        output = self.predict(output)
        # print(output)
        # output = torch.sigmoid(output)
        output = torch.squeeze(output,1)
        return output

    def fit(self, user_input, item_input, metapath_inputs):

        # user_input.shape/item_input.shape: batch_size, negative_num + 1
        # metapath_inputs: num_of_metapath_types, batch_size, negative_num + 1, path_num, latent_dim, hop_num
        path_output = None
        for i in range(len(metapath_inputs)):
            
            output = self.path_latent_vecs[i](tuple(metapath_inputs[i]))

            if path_output is None:
                path_output = output
            else:

                path_output = torch.cat((path_output, output), 2)
        # print('before path output',path_output.size())
        # batch_size * negative_num_plus, latent_dim, metapath_type
        # path_output = path_output.view((len(metapath_inputs), -1, self.latent_dim))
        path_output = path_output.view((-1,len(metapath_inputs), self.latent_dim))
        # print('after path output',path_output.size())
        user_input = self.user_latent(user_input)
        item_input = self.item_latent(item_input)

        path_atten = self.metapath_att(user_input, item_input, path_output)
        #set checkpoint
        # user_atten = checkpoint(self.user_att(user_input, path_atten))
        # item_atten = checkpoint(self.item_att(item_input, path_atten))
        user_atten = self.user_att(user_input, path_atten)
        item_atten = self.item_att(item_input, path_atten)

        output = torch.cat((user_atten, path_atten, item_atten), 1)
        # print(output)
        for layer in self.layers:
            output = layer(output)
            output = F.relu(output)
        
        output = self.predict(output)
        # print(output)
        # output = torch.sigmoid(output)
        output = torch.squeeze(output,1)
        return output, user_atten

