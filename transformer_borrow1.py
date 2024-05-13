from torch import Tensor
import torch.nn.functional as f
from torch import nn
import torch
# Q,K,V are batches of matrices, each shape (m,vertices,channel) ~ (m,v,c)
# bmm means Batch Matrix-Matrix product : (m,v,c) * (m,c,t) -> (m,v,t)
def scaled_dot_product_attention(query:Tensor,key: Tensor,value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1,2)) #(m,v,c) * (m,c,v) -> (m,v,v)
    scale = query.size(-1) ** 0.5 # just a scalar , the square root of the depth of the last dimension = c
    softmax = f.softmax(temp/scale,dim=2) # temp/scale have shape (m,v,v)
                                           # f.softmax(temp/scale,dim=-1) have shape (m,v,v), but all the value are softmax along the last dim
                                           # so softmax have shape (m,v,v)
    return softmax.bmm(value) # (m,v,v) * (m,v,c) -> (m,v,c) -> same shape as Q,K,V
class AttentionHead(nn.Module):
    def __init__(self,dim_in:int, dim_qkv:int):
        super().__init__()
        self.q = nn.Linear(dim_in,dim_qkv) # 1-d convolution is no diffrent than Fully connected
        self.k = nn.Linear(dim_in,dim_qkv)
        self.v = nn.Linear(dim_in,dim_qkv)
    def forward(self,query: Tensor,key: Tensor,value: Tensor) -> Tensor:
        # actually at the very first layer, the input params query,key,value are all original tensor
        # then those are pass through 3 separate Linear layer to achieve query,key,value 
        gen_q = self.q(query) # query is m,v,dim_in -> gen_q is m,v,dim_qkv
        gen_k = self.k(key) # key is m,v,dim_in     -> gen_k is m,v,dim_qkv
        gen_v = self.v(value) # value is m,v,dim_in -> gen_v is m,v,dim_qkv
        return scaled_dot_product_attention(gen_q,gen_k,gen_v)
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads:int,dim_in: int,dim_qkv: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in=dim_in,dim_qkv=dim_qkv) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_qkv,dim_in) #m,v,num_heads*dim_qkv -> m,v,dim_in
    def forward(self,query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query,key,value) for h in self.heads],dim=2) 
            # h(query,key,value) have the shape of m,v,dim_qkv
            # we stack many h(query,key,value) on top of each other, 
            # along the 2nd dim ( also the last dim = the dim of channel)
            # so the final shape will be (m,v,num_heads*dim_qkv)
        )
        # after self.linear : #m,v,num_heads*dim_qkv -> m,v,dim_in
def positional_encoding(dim_v:int,dim_c:int,device: torch.device = torch.device("gpu:0"))->Tensor:
    pos = torch.arange(dim_v,dtype=torch.float,device=device).reshape(1,-1,1)
    # create a 1D tensor range from 0 to dim_v-1 (dim_v is the number of vertices)
    # then transform the 1D tensor (dim_v,) -> (1,dim_v,1)
    dim = torch.arange(dim_c,dtype=torch.float,device=device).reshape(1,1,-1)
    # create a 1D tensor range from 0 to dim_c-1 (dim_c is the number of channel)
    # then transform into tensor (dim_c,) -> (1,1,dim_c)
    phase = pos /(1e4**(dim/dim_c)) # dividing (1*dim_v*1) with (1,1,dim_c)
                                    # obtain (1*dim_v*dim_c)

    return torch.where(dim.long()%2==0,torch.sin(phase),torch.cos(phase))
    # return a torch tensor (1*dim_v*dim_c), with all values clamped between -1 and 1
    # select the even indexes at the dimension 2  (dimension of channel), apply the sin function to the elements
    # select the odd indexes at the dimension 2 (dimension of channel), apply the cos function
def feed_forward(dim_channel: int = 2, dim_feedforward: int = 2048) ->nn.Module:
    return nn.Sequential(
        nn.Linear(dim_channel,dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward,dim_channel)
    )
    # so the input is m*v*c , the output is still m*v*c
class Residual(nn.Module):
    def __init__(self,sublayer:nn.Module,dim_c:int=-1,dropout: float=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim_c)