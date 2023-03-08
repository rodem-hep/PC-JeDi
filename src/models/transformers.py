"""Some classes to describe transformer architectures."""

import math
from typing import Mapping, Optional, Union

import torch as T
import torch.nn as nn
from torch.nn.functional import dropout, softmax

from .modules import DenseNetwork


def merge_masks(
    q_mask: Union[T.BoolTensor, None],
    kv_mask: Union[T.BoolTensor, None],
    attn_mask: Union[T.BoolTensor, None],
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding
    information."""

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = T.full(q_shape[:-1], True, device=device)
        if kv_mask is None:
            kv_mask = T.full(k_shape[:-1], True, device=device)
        merged_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

    # If attention mask exists, create
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    return merged_mask


def attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    dim_key: int,
    attn_mask: Optional[T.BoolTensor] = None,
    attn_bias: Optional[T.Tensor] = None,
    drp: float = 0.0,
    training: bool = True,
) -> T.Tensor:
    """Apply the attention using the scaled dot product between the key query
    and key tensors, then matrix multiplied by the value.

    Note that the attention scores are ordered in recv x send, which is the opposite
    to how I usually do it for the graph network, which is send x recv

    We use masked fill -T.inf as this kills the padded key/values elements but
    introduces nans for padded query elements. We could used a very small number like
    -1e9 but this would need to scale with if we are using half precision.

    Args:
        query: Batched query sequence of tensors (b, h, s, f)
        key: Batched key sequence of tensors (b, h, s, f)
        value: Batched value sequence of tensors (b, h, s, f)
        dim_key: The dimension of the key features, used to scale the dot product
        attn_mask: The attention mask, used to blind certain combinations of k,q pairs
        attn_bias: Extra weights to combine with attention weights
        drp: Dropout probability
        training: If the model is in training mode, effects the dropout applied
    """

    # Perform the matrix multiplication
    scores = T.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)

    # Add the bias terms if present
    if attn_bias is not None:  # Move the head dimension to the first
        scores = scores + attn_bias.permute(0, 3, 1, 2)

    # Mask away the scores between invalid elements in sequence
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.unsqueeze(-3), -T.inf)

    # Apply the softmax function per head feature
    scores = softmax(scores, dim=-1)

    # Kill the nans introduced by the padded query elements
    scores = T.nan_to_num(scores, 0)

    # Apply dropout to the attention scores
    scores = dropout(scores, p=drp, training=training)

    # Finally multiply these scores by the output
    scores = T.matmul(scores, value)

    return scores


class MultiHeadedAttentionBlock(nn.Module):
    """Generic Multiheaded Attention.

    Takes in three sequences with dim: (batch, sqeuence, features)
    - q: The primary sequence queries (determines output sequence length)
    - k: The attending sequence keys (determines incoming information)
    - v: The attending sequence values

    In a message passing sense you can think of q as your receiver nodes, v and k
    are the information coming from the sender nodes.

    When q == k(and v) this is a SELF attention operation
    When q != k(and v) this is a CROSS attention operation

    ===

    Block operations:

    1) Uses three linear layers to project the sequences.
    - q = q_linear * q
    - k = k_linear * k
    - v = v_linear * v

    2) Outputs are reshaped to add a head dimension, and transposed for matmul.
    - features = model_dim = head_dim * num_heads
    - dim becomes: batch, num_heads, sequence, head_dim

    3) Passes these through to the attention module (message passing)
    - In standard transformers this is the scaled dot product attention
    - Also takes additional dropout layer to mask the attention

    4) Flatten out the head dimension and pass through final linear layer
    - results are same as if attention was done seperately for each head and concat
    - dim: batch, q_seq, head_dim * num_heads
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
        drp: float = 0,
    ) -> None:
        """
        Args:
            model_dim: The dimension of the model
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
            drp: The dropout probability used in the MHA operation
        """
        super().__init__()

        # Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        # Initialise the weight matrices
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
        self.drp = drp

    def forward(
        self,
        q: T.Tensor,
        k: Optional[T.Tensor] = None,
        v: Optional[T.Tensor] = None,
        q_mask: Optional[T.BoolTensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        attn_bias: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """
        Args:
            q: The main sequence queries (determines the output length)
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
            attn_bias: Extra bias term for the attention matrix (eg: edge features)
        """

        # If only q and q_mask are provided then we automatically apply self attention
        if k is None:
            k = q
            if kv_mask is None:
                kv_mask = q_mask
        v = v if v is not None else k

        # Store the batch size, useful for reshaping
        b_size, seq, feat = q.shape

        # Work out the masking situation, with padding, no peaking etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        # Generate the q, k, v projections, break final head dimension in 2
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(shape)
        k = self.k_linear(k).view(shape)
        v = self.v_linear(v).view(shape)

        # Transpose to get dimensions: B,H,Seq,HD (required for matmul)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate the new sequence values, for memory reasons overwrite q
        q = attention(
            q,
            k,
            v,
            self.head_dim,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            drp=self.drp,
            training=self.training,
        )  # Returned shape is B,H,Q_seq,HD

        # Concatenate the all of the heads together to get shape: B,Seq,F
        q = q.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        # Pass through final linear layer
        q = self.out_linear(q)

        return q


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style
    arcitecture.

    We choose Normformer as it has often proved to be the most stable to train
    https://arxiv.org/abs/2210.06423
    https://arxiv.org/abs/2110.09456

    It contains:
    - Multihead(self)Attention block
    - A dense network

    Layernorm is applied before each operation
    Residual connections are used to bypass each operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimensio of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(model_dim, **mha_config)
        self.dense = DenseNetwork(
            model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config
        )

        # The normalisation layers (lots from NormFormer)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.norm2(
            self.self_attn(
                self.norm1(x), q_mask=mask, attn_mask=attn_mask, attn_bias=attn_bias
            )
        )
        x = x + self.dense(self.norm3(x), ctxt)
        return x


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final
    normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)


class FullTransformerEncoder(nn.Module):
    """A transformer encoder with added input and output embedding networks.

    Sequence -> Sequence
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        edge_dim: int = 0,
        ctxt_dim: int = 0,
        te_config: Optional[Mapping] = None,
        node_embd_config: Optional[Mapping] = None,
        outp_embd_config: Optional[Mapping] = None,
        edge_embd_config: Optional[Mapping] = None,
        ctxt_embd_config: Optional[Mapping] = None,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of of the final output vector
            edge_dim: Dim. of the input edge features
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            te_config: Keyword arguments to pass to the TVE constructor
            node_embd_config: Keyword arguments for node dense embedder
            outp_embd_config: Keyword arguments for output dense embedder
            edge_embd_config: Keyword arguments for edge dense embedder
            ctxt_embd_config: Keyword arguments for context dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim
        te_config = te_config or {}
        node_embd_config = node_embd_config or {}
        outp_embd_config = outp_embd_config or {}
        edge_embd_config = edge_embd_config or {}

        # Initialise the context embedding network (optional)
        if self.ctxt_dim:
            self.ctxt_emdb = DenseNetwork(
                inpt_dim=self.ctxt_dim,
                **ctxt_embd_config,
            )
            self.ctxt_out = self.ctxt_emdb.outp_dim
        else:
            self.ctxt_out = 0

        # Initialise the TVE, the main part of this network
        self.te = TransformerEncoder(**te_config, ctxt_dim=self.ctxt_out)
        self.model_dim = self.te.model_dim

        # Initialise all embedding networks
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

        # Initialise the edge embedding network (optional)
        if self.edge_dim:
            self.edge_embd = DenseNetwork(
                inpt_dim=self.edge_dim,
                outp_dim=self.te.layers[0].self_attn.num_heads,
                ctxt_dim=self.ctxt_out,
                **edge_embd_config,
            )

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        if self.edge_dim:
            attn_bias = self.edge_embd(attn_bias, ctxt)
        x = self.node_embd(x, ctxt)
        x = self.te(x, mask=mask, ctxt=ctxt, attn_bias=attn_bias, attn_mask=attn_mask)
        x = self.outp_embd(x, ctxt)
        return x
