import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#缩放点积注意力
class SDPAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,q,k,v,mask=None):
        dk = q.size(-1)
        qk = torch.matmul(q,k.transpose(-1,-2))/np.sqrt(dk)
        
        if mask is not None:
            qk = qk.masked_fill(mask==0,value=-1e4)
        attn_weight = F.softmax(qk,dim=-1)
        output = torch.matmul(qk,v)     #[B,H,seq_len_q,embed_v]
        return output,attn_weight
    

#多头注意力
class MHAttention(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim//self.num_heads
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.sdpattention = SDPAttention()

    def forward(self,x,mask=None):
        batch_size,seq_len,_ = x.shape()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)


        #mask形状为[batch_size,seq_len]
        if len(mask.size()) == 2:       #[batch_size,seq_len]
            mask = mask.unsqeeze(1).unsqueeze(3)     #[batch_size,1,seq_len,1]
        elif len(mask.size()) == 3:     #[batch_size,seq_len1,seq_len2]
            mask = mask.unsqueeze(1)    #[batch_size,H,seq_len1,seq_len2]

        output,_ = self.sdpattention(q,k,v,mask)   

        #output = output.reshape(batch_size,self.num_heads,-1,self.embed_dim)
        output = output.transpose(1,2).reshape(batch_size,-1,self.embed_dim)
        output = self.o_proj(output)
        return output

#Multi-Query Attention
class MQAttention(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim//self.num_heads
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.head_dim)
        self.sdpattention = SDPAttention()
        self.o_proj = nn.Linear(self.embed_dim,self.embed_dim)

    def forward(self,x,mask=None):
        batch_size,seq_len,_ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)   #[batch_size,num_heads,seq_len,head_dim]
        k = k.unsqueeze(1)  #[batch_size,1,seq_len,head_dim]
        v = v.unsqueeze(1)  #[batch_size,1,seq_len,head_dim]

        if len(mask.size()) == 2:       #[batch_size,seq_len]
            mask = mask.unsqeeze(1).unsqueeze(3)     #[batch_size,1,seq_len,1]
        elif len(mask.size()) == 3:     #[batch_size,seq_len1,seq_len2]
            mask = mask.unsqueeze(1)    #[batch_size,H,seq_len1,seq_len2]
        output,_ = self.sdpattention(q,k,v,mask).transpose(1,2)
        output = output.reshape(batch_size,-1,self.embed_dim)
        output = self.o_proj(output)
        return output 
        


#Grouped-Query Attention
class GQAttention(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8,num_groups:int=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = self.embed_dim // self.num_heads
        self.group_dim = self.head_dim * self.num_groups
        self.num_group_heads = self.num_heads // self.num_groups
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.group_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.group_dim)
        self.o_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.sdpattention = SDPAttention()

    def forward(self,x,mask=None):
        batch_size,seq_len,_ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(batch_size,seq_len,self.num_groups,self.head_dim).transpose(1,2)
        v = v.reshape(batch_size,seq_len,self.num_groups,self.head_dim).transpose(1,2)
        k = k.repeat(1,self.num_group_heads,1,1) #[B,H,seq_len,head_dim]
        v = v.repeat(1,self.num_group_heads,1,1)

        if len(mask.size()) == 2:       #[batch_size,seq_len]
            mask = mask.unsqeeze(1).unsqueeze(3)     #[batch_size,1,seq_len,1]
        elif len(mask.size()) == 3:     #[batch_size,seq_len1,seq_len2]
            mask = mask.unsqueeze(1)    #[batch_size,H,seq_len1,seq_len2]

        output,_ = self.sdpattention(q,k,v,mask).transpose(1,2)
        output = output.reshape(batch_size,-1,self.embed_dim)
        output = self.o_proj(output)
        return output   



#稀疏注意力
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()

#交叉注意力
class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
