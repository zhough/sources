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
        output = torch.matmul(qk,v)
        return output,attn_weight
    

#多头注意力
class MHAttention(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim/self.num_heads
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

        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        k = k.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        v = v.reshape(batch_size,seq_len,self.num_heads,self.head_dim)

        q = q.reshape(-1,seq_len,self.head_dim)
        k = k.reshape(-1,seq_len,self.head_dim)
        v = v.reshape(-1,seq_len,self.head_dim)
        
        output,_ = self.sdpattention(q,k,v,mask)
        return output

#Multi-Query Attention
class MQAttention(nn.Module):
    def __init__(self):
        super().__init__()

#Grouped-Query Attention
class GQAttention(nn.Module):
    def __init__(self):
        super().__init__()

#稀疏注意力
class SparseAttention(nn.Module):
    def __init__(self):
        super().__init__()