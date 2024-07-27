'''
Author: jing xu
Date: 2023-04-13 15:30:32
LastEditors: jing xu
LastEditTime: 2024-03-12 20:49:56
FilePath: /df_detr_relation/code/relation.py
'''

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from aocn.predictor import MLP
import numpy as np
from aocn.general import (
    get_clones,
    get_activation_fn,
    get_proposal_pos_embed,
)


class Relation(nn.Module):
    def __init__(self,k=1,hidden_dim = 256,feature_levels = 1,dim_feedforward = 1024,nhead=8,dropout=0.1,activation="relu",base_type="Transformer"):
        super().__init__()
        self.base_type = base_type #baseline的类型, "CNN" "Transformer"
        self.k = k
        self.hidden_dim = hidden_dim
        self.feature_levels = feature_levels

        #这个是cnn型feature map输入才需要的
        # self.proposal_project = MLP(49,128,1,3)

        #GCFL
        # Feature Map Update
        self.memory_project1 = MLP(hidden_dim*10,dim_feedforward,hidden_dim,3) #拼接切片特征后上/下采样
        self.memory_project2 = MLP(hidden_dim*2, dim_feedforward,hidden_dim,3)

        self.conv1 = nn.Conv2d(hidden_dim,dim_feedforward,3,2,1)
        self.conv2 = nn.Conv2d(dim_feedforward,hidden_dim,1) #1x1卷积改变维度
        self.up = nn.UpsamplingBilinear2d(scale_factor=2) #good
        self.conv3 = nn.Conv2d(hidden_dim*2,hidden_dim,1) #1x1卷积改变维度


        # query update
        decoder_layer = SelfAttentionDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
        self.new_decoder = NewDecoder(decoder_layer,4)



    
    # def forward(self,all_slice_base_out,all_slice_base_feature_map,all_slice_base_proposal_feature,
    #             feature_shape,start_index):
    def forward(self,all_slice_base_out,all_slice_base_feature_map,all_slice_base_proposal_feature,feature_shape):
        """
        切片关系学习模块

        Args:
            all_slice_base_out ([[batch1_all_slice_base_out][batch2_all_slice_base_out]]): baseline 输出的预测信息
                1. batch_all_slice_base_out : [{slice1_base_out}{slice2_base_out}{slice3_base_out}...]
                2. slice_base_out{'pred_bbox'} : [all_queries,4]


            all_slice_base_feature_map ([[batch1_all_slice_base_feature_map][batch2_all_slice_base_feature_map]]):baseline输出的特征图
                1. batch_all_slice_base_feature_map : [[slice1_feature_map][slice2_feature_map]...] 
                2. slice_feature map : [[h1,w1][h2,w2][h3,w3]] 如果是包含FPN的feature map 的话

            all_slice_base_proposal_feature ([[batch1_all_slice_base_proposal_feature][batch2_all_slice_base_proposal_feature]]): baseline输出的proposal feature
                1. 在transformer - based model里就是query, 在CNN based model里就是proposal feature
                2. batch_all_slice_base_proposal_feature : [[slice1_proposal_feature][slice2_proposal_feature]]
                3. slice_base_proposal_feature : [all_queries,hidden_dim]
        """

        if self.base_type == "Transformer":
            #1. 取topk objects and object feature
            all_slice_topk_bbox, all_slice_k_index, batch_slice_num = self.topk_bbox_and_idx(all_slice_base_out)
            all_slice_topk_p_features = self.extract_features(
                all_slice_base_proposal_feature, all_slice_k_index)  # [slice,k,dim]
            #2. 根据bbox, 生成pos_embed (后面decoder要用)
            all_slice_pos_embed = self.proposal_pos_embed(all_slice_topk_bbox)

            # #3. 对于transformer-based memory 可以先做了上下采样到10, 后面在转换成h w的来做cnn
            sampled_feature_map = self.memory_sample(all_slice_base_feature_map) #GCFL里的前面部分的上/下采样和MLP
            
            # #4. memory_format to cnn_format
            sampled_feature_map = self.featureMap_Format(sampled_feature_map,feature_shape)

            #因为传进来的memory也是h,w型, 所以用cnn型memorysample #如果传进来的feature map是h w型的, 就得用这个
            # sampled_feature_map,all_slice_feature_map = self.cnn_memory_sample(all_slice_base_feature_map) ### Memory Ablation, 但是因为需要all_slice_feature_map所以保留
        if self.base_type == "CNN":
            # 1. 取出每个切片的topk 预测框的数据
            # all_slice_topk_bbox , all_slice_k_index ,batch_slice_num = self.topk_bbox_and_idx(all_slice_base_out)
            # all_slice_topk_p_features = self.extract_features(all_slice_base_proposal_feature,all_slice_k_index) #[slice,k,dim]
            all_slice_topk_bbox = self.cnn_out_2_relation(all_slice_base_out) #[38,k,4]
            all_slice_pos_embed = self.proposal_pos_embed(all_slice_topk_bbox)
            batch_slice_num = [len(batch) for batch in all_slice_base_out]
            all_slice_topk_p_features = self.cnn_proposal_feature_2_relation(all_slice_base_proposal_feature)#[38, 1, 256]
            # batchsize = len(batch_slice_num)

            # # 2. 多切片Feature map合并
            feature_maps,all_slice_feature_map = self.cnn_memory_sample(all_slice_base_feature_map) # 
            # all_feature_map = self.memory_sample(all_slice_base_feature_map) #[bs,all_pixiels,dim]

            # #把feature map弄成[bs,dim,h,w]的形式, 方便后续用卷积
            # feature_maps = self.featureMap_Format(all_feature_map,feature_shape,start_index)

        
        #5. 特征学习 CNN   #GCFL里卷积和上采样的部分
        for level,f_map in sampled_feature_map.items(): #对每个level的feature map分别学习
            temp = F.leaky_relu(self.conv1(f_map)) #[bs,1024,h/2,w/2]
            temp = self.conv2(temp) #[bs,256,h/2,w/2]
            temp = self.up(temp) ##[bs,256,h,w]
            # 如果原来的h, w是奇数, 那么经过/2 再*2之后是偶数, 这俩维度对不上, 所以要加一个padding
            diff_y = f_map.size()[2] - temp.size()[2]
            diff_x = f_map.size()[3] - temp.size()[3]

            temp = F.pad(temp,[diff_x//2,diff_x-diff_x//2,
                            diff_y//2,diff_y-diff_y//2])

            temp = torch.cat((f_map,temp),dim=1)#bs,channel,h,w 在channel维度拼接 [bs,512,h,w]
            ##
            temp = self.conv3(temp)#[bs,256,h,w]
            sampled_feature_map[level] = temp
            
        #6.将cnn型feature_map [bs,dim,h,w] 整形成[bs,all_pixiels,dim], 因为是后面decoder cross-attention的输入格式
        case_feature = self.featureMap_Format_Reverse(sampled_feature_map)#[bs,all_pixels,dim]
        
        #7. 循环每个slice单独进行new decoder的计算
        batch_start = 0
        max_slices = np.array(batch_slice_num).max()
        batchsize = len(batch_slice_num)
        dim = self.hidden_dim
        all_aligned_query = torch.zeros((batchsize,max_slices+1,max_slices+1,dim)).to(all_slice_topk_p_features.device) #补0填充query , +1是因为后面query还会拼接一个原始未更新过的query
        all_aligned_ref_windows = torch.zeros(
            (batchsize, max_slices+1, max_slices+1, 4)).to(all_slice_topk_p_features.device)
        for bs,bt_slice_num in enumerate(batch_slice_num):#每个case（batch)
            cur_case_feature = case_feature[bs].unsqueeze(0) #[1,all_pixels,dim] #GCFL得到的当前case的feature
            case_queries = all_slice_topk_p_features[batch_start:batch_start+bt_slice_num,...].permute(1,0,2)#[k,case_slice_num,dim] 当前case所有的从base detector得到的object features
            case_pos = all_slice_pos_embed[batch_start:batch_start+bt_slice_num,...].permute(1,0,2)#对应的每个bbox的pos_embeds
            case_ref_windows = all_slice_topk_bbox[batch_start:batch_start+bt_slice_num,...].permute(1,0,2)#base detector得到的对应的bbox
            for i in range(bt_slice_num):#每个slice
                current_slice_memory = all_slice_base_feature_map[bs][i]
                # current_slice_memory = all_slice_feature_map[bs][i] #memory是h,w型用这个
                memory = torch.cat((current_slice_memory,cur_case_feature),dim=-1) #[1,all_pixels,hidden_dim*2]
                memory = self.memory_project2(memory)
                current_query = case_queries[0][i].unsqueeze(0) #用来后面保留原始query [1,dim]
                current_ref_window = case_ref_windows[0][i].unsqueeze(0)
                # current_output,roi = self.new_decoder(
                #     case_queries, case_pos, memory, scale_shape, src_mask, start_index, valid_ratios, case_ref_windows)
                # current_output, roi = self.new_decoder(
                #     case_queries, case_pos, current_slice_memory, scale_shape, src_mask, start_index, valid_ratios, case_ref_windows)
                # current_output, roi = self.new_decoder(case_queries, case_pos, current_slice_memory) #tgt,query_pos,memory  ### 做GCFL Ablation时用这个
                current_output, roi = self.new_decoder(case_queries, case_pos, memory) #tgt,query_pos,memory 
                
                # current_output.shape [decode_layers_num,1,slice_nums,dim]
                current_slice_queries = torch.cat((current_query,current_output[-1][0]),dim=0) #[case_slice_num+1,dim]
                current_slice_ref_windows = torch.cat((current_ref_window,case_ref_windows[0]),dim=0) # [case_slice_num+1,4]
                all_aligned_query[bs][i][:bt_slice_num+1,...] = current_slice_queries
                all_aligned_ref_windows[bs][i][:bt_slice_num+1,...] = current_slice_ref_windows
            batch_start += bt_slice_num
        #all_aligned_query.unsqueeze(0).shape   [1, 8, 17, 17, 256]
        #all_aligned_ref_windows.shape   [1,8, 17, 17, 4]  #前面的1是给多少层decoder输出特征
        return all_aligned_query.unsqueeze(0), all_aligned_ref_windows.unsqueeze(0)

    def cnn_proposal_feature_2_relation(self,all_slice_base_proposal_feature):
        #return : [all_slices,k,dim]
        for bs,batch in enumerate(all_slice_base_proposal_feature):
            for idx,slice in enumerate(batch): #slice : [1,256,7,7]
                if idx == 0:
                    case_proposals = slice
                else:
                    case_proposals = torch.cat((case_proposals,slice),dim=0) 
            
            #256,7,7 -> 256,1
            case_proposals = self.proposal_project(case_proposals.view(len(batch),self.hidden_dim,-1)) #[slice_num,256,1]

            if bs==0:
                all_proposal_feature = case_proposals
            else:
                all_proposal_feature = torch.cat((all_proposal_feature,case_proposals),dim=0)
        
        return all_proposal_feature.permute(0,2,1) #[all_slices,1,256]

            

    def cnn_memory_sample(self,feature_map):
        #feature_map_dict['0'].shape [bs,channels,h,w]
        sampled_feature_map = OrderedDict()
        all_slice_feature_map =[]
        for bs,batch in enumerate(feature_map):
            temp_dict = OrderedDict()
            temp_case_feature_map = []
            for idx,slice in enumerate(batch): 
                for i,(k,v) in enumerate(slice.items()):
                    if i ==0:
                        slice_fmap = v.view(1,256,-1) 
                    else:
                        slice_fmap = torch.cat((slice_fmap,v.view(1,256,-1)),dim=-1)
                    if idx == 0 and k not in temp_dict.keys():#第一个切片的所有level都来
                        temp_dict[k] = v  #[1,256,12,8]
                    else:#不是第一个切片
                        temp_dict[k] = torch.cat((temp_dict[k],v),dim=0)
                
                temp_case_feature_map.append(slice_fmap.permute(0,2,1))
            
            ### Memory Ablation  AnchorDETR_AOCN在做no_memory ablation的时候需要上面的部分, 不需要下面的部分

            #一个case结束 每个level的feature map都要上下采样
            for k,v in temp_dict.items():
                s_num,dim,h,w = v.shape
                v = v.view(s_num,dim,-1)
                pixels = v.shape[-1]
                v = F.interpolate(v.unsqueeze(0).unsqueeze(0),size=(10,dim,pixels),mode="nearest").squeeze(0) #[1,10,dim,pixels]
                v = v.permute(0,3,1,2).view(1,pixels,-1) #[1,pixels,10*dim]
                v = self.memory_project1(v).permute(0,2,1) #[1,dim,pixles]   10*dim -> dim
                v = v.view(1,dim,h,w)
                temp_dict[k] = v

                if bs ==0 and k not in sampled_feature_map.keys():
                    sampled_feature_map[k] = temp_dict[k]  #[1,dim,h,w]
                else:
                    sampled_feature_map[k] = torch.cat((sampled_feature_map[k],temp_dict[k]),dim=0)
            
            ### Memory Ablation
            
            all_slice_feature_map.append(temp_case_feature_map)
        
        # {'0': [bs,dim,h,w],'1':}
        return sampled_feature_map,all_slice_feature_map

    def cnn_out_2_relation(self,out):
        #ourput : [all_batch_slice_num,k,4] all_batch_slice_num 所有batch的slice都放到第一个bs维度里
        all_slice_base_out = []
        for bs,batch in enumerate(out):
            for idx,slice in enumerate(batch):
                if(bs ==0 and idx == 0):
                    all_slice_base_out = slice[0][0][:4].unsqueeze(0).unsqueeze(0)
                else:
                    all_slice_base_out = torch.cat((all_slice_base_out,slice[0][0][:4].unsqueeze(0).unsqueeze(0)),dim=0)
        
        return all_slice_base_out


    def topk_bbox_and_idx(self, out):
        # topk 的下标索引
        '''
        Input:
            out : [[{},{},...][{},{},...],...] 
                len(out) = bs , len(out[0]) = case1_slice_num , 
                out[0][0]['pred_boxes']: [1,query_num,4]
                out[0][0]['pred_logits'] : [1,query_num,num_class+1] 下标1代表appendix

        output:
        * all_slice_index : [all_batch_slice_num,k]
        * topk_boxes : [all_batch_slice_num,k,4]
        * batch_slice_num : [batch1_slice_num,batch2_slice_num,...]
        '''
        all_slice_index = [] #[slice_num,k]
        batch_slice_num = [len(batch) for batch in out]
        
        for bs,batch in enumerate(out):
            for idx,slice in enumerate(batch):
                _,slice_index = torch.topk(slice['pred_logits'][...,1], k = self.k , dim=-1)
                slice_index = slice_index.squeeze(0).data
                all_slice_index.append(slice_index)
                if bs==0 and idx == 0:
                    topk_boxes = slice['pred_boxes'][0][slice_index,...].unsqueeze(0) #[slices,k,4]
                else:
                    topk_boxes = torch.cat(
                        (topk_boxes, slice['pred_boxes'][0][slice_index, ...].unsqueeze(0)), dim=0)  # [slice,k,4]
                
        
        return topk_boxes,all_slice_index,batch_slice_num
    
    def extract_features(self, proposals_features, index):
        #input:
        # proposals_features  [slice_num , num_layers,bs,num_queries,dim]
        #  proposals_features[0].shape [num_layers,bs,num_queries,dim]
        # index : [slice_num ,k ]
        # batch_slice_num:
        # output : [all_batch_slice_num,k,dim]
        #根据topk的index 取出对应的proposals_features
        before_slices = 0
        for bs,batch in enumerate(proposals_features):

            slice_num = len(batch)
            batch_index = index[before_slices:(before_slices+slice_num)]
            for sli,slice in enumerate(batch):
                slice_proposal_feature = slice[-1][0][batch_index[sli],...] #[k,dim]
                if bs==0 and sli==0:
                    topk_proposals_features = slice_proposal_feature.unsqueeze(0)
                else:
                    topk_proposals_features = torch.cat((topk_proposals_features,slice_proposal_feature.unsqueeze(0)),dim=0)

            before_slices += slice_num

        #topk_p_features.shape [all_batch_slices,k,dim]
        return topk_proposals_features
    
    
    def proposal_pos_embed(self,topk_bbox):
        # 根据proposal bbox 生成包含pos 和 size 的 pos_embed
        pos = get_proposal_pos_embed(
            topk_bbox[..., :2], self.hidden_dim)  # [all_batch_slices,k,256]
        size = get_proposal_pos_embed(
            topk_bbox[..., 2:], self.hidden_dim)  # [all_batch_slices,k,256]
        pos_embed = pos + size  # 位置信息包含位置和size
        
        return pos_embed
    
    def memory_sample(self,all_slice_base_memory):
        #input all_slice_base_memory : length slice_num 
        #all_slice_base_memory[0].shape [bs,all_pixels,dim] #test :

        #output: [sample_slice_num,all_pixels,dim]
        for bs,batch in enumerate(all_slice_base_memory):
            
            all_memory = torch.cat(batch,dim=0).unsqueeze(0).unsqueeze(0) #[1,1,all_slice_num,all_pixels,dim]
            all_pixels,dim = all_memory.shape[-2:]

            #下采样or上采样
            all_memory = F.interpolate(all_memory,size=[10,all_pixels,dim],mode="nearest")
            all_memory = all_memory.squeeze(0).squeeze(0)
            all_memory = all_memory.permute(1,0,2).reshape(all_pixels,-1).contiguous()
            all_memory = self.memory_project1(all_memory).unsqueeze(0) #[1,all_pixels,dim]
            if bs == 0:
                all_slice_memory = all_memory
            else:
                all_slice_memory = torch.cat((all_slice_memory,all_memory),dim=0) #[bs,all_pixels,dim]
        return all_slice_memory
    
    def featureMap_Format(self,all_feature_map,feature_shape):
        """
        对于从transformer过来的memory, 要转变一下格式, 变成[h1,w1][h2,w2][h3,w3]

        Args:
            all_slice_base_feature_map (_type_): transformer encoder输出的memory
            feature_shape : [feature_levels,2]  每层feature map 的 h w 
        
        Output:
            {'3' : 3rd_level_all_slice_base_feature_map,'2':2nd_level_all_slice_base_feature_map,...}
        """
        feature_map_dict = {}  #key越小, 代表越大的feature map
        feature_levels = feature_shape.shape[0]
       
        ##通过feature_shape生成start_index
        start_index = [0]
        before_accu = start_index[0] 
        if feature_levels > 1 :
            for cur_shape in feature_shape:
                h = cur_shape[0]
                w = cur_shape[1]
                start_index.append(h*w + before_accu) #正好hw就是下一次开始的位置
                before_accu = start_index[-1]

            start_index = start_index[:-1]
        
        
        for bs,case_feature in enumerate(all_feature_map):
            for feature_level,slice_shape in enumerate(feature_shape):
                if(feature_level == feature_levels -1 ):
                    #最后一个
                    temp_feature = case_feature[start_index[feature_level]:]
                else:
                    temp_feature = case_feature[start_index[feature_level]:start_index[feature_level+1]]
                temp_feature = temp_feature.view(slice_shape[0],slice_shape[1],-1).permute(2,0,1) # [h w dim] -> [dim(channel),h,w]
                if(str(feature_level) not in feature_map_dict.keys()):
                    feature_map_dict[str(feature_level)] = temp_feature.unsqueeze(0) # [1,h,w,dim] 第一个维度用来拼接bs
                else:
                    feature_map_dict[str(feature_level)]  = torch.cat((feature_map_dict[str(feature_level)],temp_feature.unsqueeze(0)),dim=0)
        #feature_map_dict['0'].shape [bs,channels,h,w]
        return feature_map_dict
    
    def featureMap_Format_Reverse(self,feature_map_dict):
        """
        再把feature map形式变成transformer decoder要的memory的形式

        Args:
            feature_map_dict (_type_): _description_
        
        Output:
            memory : [bs,all_pixles,dim]
        """
        feature_levels = len(feature_map_dict)
        for i in range(feature_levels):
            if(i == 0):
                memory = torch.flatten(feature_map_dict[str(i)],2) #[bs,dim,pixles_num]
            else:
                memory = torch.cat((memory,torch.flatten(feature_map_dict[str(i)],2)),dim=-1)
        
        return memory.permute(0,2,1) #[bs,all_pixels,dim]
    
    def featureMap_cnn_feature_map(self,feature_map_dict):
        """
        再把feature map形式变成transformer decoder要的memory的形式

        Args:
            feature_map_dict (_type_): _description_
        
        Output:
            memory : [bs,all_pixles,dim]
        """
        feature_levels = len(feature_map_dict)
        for k,v in feature_map_dict.items():
            if(k == '0'):
                memory = torch.flatten(feature_map_dict[k],2) #[bs,dim,pixles_num]
            else:
                memory = torch.cat((memory,torch.flatten(feature_map_dict[k],2)),dim=-1)
        
        return memory.permute(0,2,1) #[bs,all_pixels,dim]
    
        

class SelfAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.multihead_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        query_pos,
        memory
    ):
        # tgt query_pos 都是[bs,num_queries,256]
        q = k = self.with_pos_embed(tgt, query_pos)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).transpose(0,1),
                                    key=memory.transpose(0, 1),
                                    value=memory.transpose(0,1))[0]
        tgt2 = tgt2.transpose(0,1)
        roi = None

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, roi  # [bs,num_queries,256] None detection没有roi
    

class NewDecoder(nn.Module):
    def __init__(self,decode_layer,num_layers):
        super().__init__()
        self.layers = get_clones(decode_layer,num_layers)
    
    def forward(self,tgt,query_pos,memory):
        output = tgt
        inter = []
        inter_roi = []

        for i, layer in enumerate(self.layers):
        # hack to return mask from the last layer
            if i == len(self.layers) - 1:
                layer.inferencing = False
                layer.multihead_attn.inferencing = False

            output, roi_feat = layer(
                output,
                query_pos,
                memory
            )#[bs,num_queries,256]
            inter.append(output)
            inter_roi.append(roi_feat)

        return torch.stack(inter), None