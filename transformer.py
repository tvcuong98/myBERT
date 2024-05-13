import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
## standalone def
def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )
class Residual(nn.Module):
    """
    : ##### case 1 : Input is just 1 tensor : 
    : Simple, just pass that tensor through a network, output a tensor same shape, and doing addition the output and input tensor
    : ##### case 2 : Input is just 3 tensor :  Passing through multihead attention
    : The 3 tensors is actually one same input tensor: src,src,src
    : After calculating multihead attention, it output a tensor with the same shape with the input
    : Do the addition : output + src

    ######### SHAPE IN BOTH CASES #########
    : input : m,len,dim
    : output : m,len,dim

    """
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class AttentionHead(nn.Module):
    """
    :########PARAM#######
    :dim_out is actually dim_qkv
    :dim_qkv is equal to dim_in if the number of heads is 1
    :dim_qkv is equal to dim_in//num_heads if the number of heads is larger than 1
    :########INPUT#######
    :input : m,v,dim_in
    :output : m,v,dim_out
    """

    def __init__(self,dim_in:int,dim_qkv:int,mask=None):
        super().__init__()
        self.q = nn.Linear(dim_in,dim_qkv)
        self.k = nn.Linear(dim_in,dim_qkv)
        self.v = nn.Linear(dim_in,dim_qkv)
        self.mask = mask
        self.dim_qkv = dim_qkv
    def scaled_dot_product_attention(self, Q, K, V):
        assert (Q.size(-1) == K.size(-1) and K.size(-1) == V.size(-1)),f"dim Q {Q.size(-1)} != dim K {K.size(-1)} or dim K {K.size(-1)} != dim V {V.size(-1)}"
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_qkv)
        if self.mask is not None:
            attn_scores = attn_scores.masked_fill(self.mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    def forward(self,query: Tensor,key: Tensor,value: Tensor) -> Tensor:
        # actually at the very first layer, the input params query,key,value are all original tensor
        # then those are pass through 3 separate Linear layer to achieve query,key,value 
        gen_q = self.q(query) # query is m,v,dim_in -> gen_q is m,v,dim_qkv
        gen_k = self.k(key) # key is m,v,dim_in     -> gen_k is m,v,dim_qkv
        gen_v = self.v(value) # value is m,v,dim_in -> gen_v is m,v,dim_qkv
        return self.scaled_dot_product_attention(gen_q,gen_k,gen_v)
class MultiHeadAttention(nn.Module):
    """
    : input : batch_size x len x dim_in
    : output: batch_size x len x dim_qkv*num_heads  
    : output -> Linear(dim_qkv*num_heads, dim_in) 
    : output -> batch_size x len x dim_in
    """
    def __init__(self, num_heads:int,dim_in:int,dim_qkv:int,mask=None):
        super(MultiHeadAttention, self).__init__()
        self.mask = mask
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in=dim_in,dim_qkv=dim_qkv,mask=self.mask) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(dim_qkv*num_heads,dim_in) # the final linear layer is actually (nn.Linear(dim_in,dim_in))
    def forward(self,query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query,key,value) for h in self.heads],dim=2) 
            # h(query,key,value) have the shape of m,v,dim_qkv
            # we stack many h(query,key,value) on top of each other, 
            # along the 2nd dim ( also the last dim = the dim of channel)
            # so the final shape will be (m,v,num_heads*dim_qkv) = (m,v,dim_in)
            # since num_heads*dim_qkv = dim_in (which is also why we have to choose dim_in is divisible by num_heads)
            # so the final linear layer is actually (nn.Linear(dim_in,dim_in))
        )
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_in: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mask=None
    ):
        super().__init__()
        dim_qkv = max(dim_in // num_heads, 1)
        self.multi_head_attn = Residual(
            MultiHeadAttention(num_heads, dim_in, dim_qkv),
            dimension=dim_in,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_in, dim_feedforward),
            dimension=dim_in,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.multi_head_attn(src, src, src)
        return self.feed_forward(src)
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mask = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout,mask)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src
    

class TransformerDecoderLayer(nn.Module):
    """
    : ###### Explaination ######:
    : Here, in the forward function, there is another param : 
    : memory : the output after the final layer of encoder
    : ###### Input ######:
    : tgt : batch_size, tgt_len, dim_in
    : memory : batch_size, src_len, dim_in 
    : ###### output ######:
    : output : batch_size, tgt_len, dim_in
    """
    def __init__(
        self,
        dim_in: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mask=None,
    ):
        super().__init__()
        dim_qkv  = max(dim_in // num_heads, 1)
        self.multi_head_attn= Residual(
            MultiHeadAttention(num_heads, dim_in, dim_qkv,mask),
            dimension=dim_in,
            dropout=dropout,
        )
        self.cross_multi_head_attn = Residual(
            MultiHeadAttention(num_heads, dim_in, dim_qkv),
            dimension=dim_in,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_in, dim_feedforward),
            dimension=dim_in,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.multi_head_attn(tgt, tgt, tgt)
        tgt = self.cross_multi_head_attn(tgt, memory, memory) # so tgt is tgt is pre-query, memory is pre-key, memory is pre-value
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mask=None
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout, mask)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)
class Transformer(nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU(),
        mask=None
    ):
        super().__init__()
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mask=src_mask
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mask=tgt_mask
        )
    def generate_mask(self, src, tgt):
        """
        : src: shape is (batch_size,max_seq_length) <- src_data (batch_size,max_seq_length)
        : src: value range from 1 -> src_vocab_size 
        : tgt: shape is (batch_size,max_seq_length-1) <- tgt_data[:,:-1] (batch_size,max_seq_length-1)
        : src: value range from 1 -> src_vocab_size  
        """
        src_mask = torch.ones(1, src.size(1), src.size(1)).bool()
        tgt_mask = torch.ones(1, tgt.size(1), tgt.size(1)).bool()
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        The positional encoding have been in encoder and decoder
        """
        
        return self.decoder(tgt, self.encoder(src))
src = torch.rand(64, 32, 512)
tgt = torch.rand(64, 16, 512)
out = Transformer()(src, tgt)
print(out.shape)
# torch.Size([64, 16, 512])