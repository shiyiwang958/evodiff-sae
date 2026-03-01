import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from esm.modules import TransformerLayer, LearnedPositionalEmbedding, ESM1bLayerNorm, AxialTransformerLayer
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_coeff = l1_coeff
        self.hidden_dim = hidden_dim

    def forward(self, x, nonpad_mask, input_mask, method="unmask"):
        z = F.relu(self.encoder(x))
        x_recon = self.decoder(z)
        nonpad_mask = nonpad_mask.permute(1, 2, 0).unsqueeze(-1) # R x C x B -> R x C x B x 1
        input_mask = input_mask.permute(1, 2, 0).unsqueeze(-1) # R x C x B -> R x C x B x 1
        # Divide by the batch size in the loss
        if method == "unmask": # only evaluate loss on unmasked positions
            sae_loss = ((x_recon - x)**2 * nonpad_mask * input_mask).sum() / (self.hidden_dim * x.shape[2]) + self.l1_coeff * (z * nonpad_mask * input_mask).abs().sum() / (self.hidden_dim * x.shape[2]) # loss evaluated on non-padded positions
        elif method == "mask": # only evaluate loss on masked positions
            sae_loss = ((x_recon - x)**2 * nonpad_mask * (1 - input_mask)).sum() / (self.hidden_dim * x.shape[2]) + self.l1_coeff * (z * nonpad_mask * (1 - input_mask)).abs().sum() / (self.hidden_dim * x.shape[2]) # loss evaluated on non-padded positions
        elif method == "all": # evaluate loss on all positions
            sae_loss = ((x_recon - x)**2).sum() / (self.hidden_dim * x.shape[2]) + self.l1_coeff * z.abs().sum() / (self.hidden_dim * x.shape[2])
        else:
            raise ValueError(f"Invalid method {method}, must be one of ['unmask', 'mask', 'all']")
        # to implement a Cross-Entropy loss
        return x_recon, sae_loss

class MSATransformerSAE(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"
    https://doi.org/10.1101/2021.02.12.430858

    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, insertion_layer, use_ckpt=False, n_tokens=len(PROTEIN_ALPHABET),
                 padding_idx=PROTEIN_ALPHABET.index(PAD), mask_idx=PROTEIN_ALPHABET.index(MASK),
                 max_positions=1024, tie_weights=True):
        super(MSATransformerSAE, self).__init__()
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        if insertion_layer < 0 or insertion_layer >= n_layers:
            raise ValueError(f"insertion_layer must be in [0, {n_layers - 1}], got {insertion_layer}")
        self.insertion_layer = insertion_layer
        
        self.layers_before = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(insertion_layer + 1)
            ]
        )
        self.sae = SparseAutoencoder(d_model, d_hidden)
        self.layers_after = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers - insertion_layer - 1)
            ]
        )
        self.padding_idx = padding_idx
        self.sae_loss = None

        # self.contact_head = ContactPredictionHead()
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        if tie_weights:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=self.embed_tokens.weight
            )
        else:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=nn.Linear(d_model, n_tokens).weight
            )

        self.use_ckpt = use_ckpt

    def forward(self, tokens, nonpad_mask, input_mask, method="unmask"):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C

        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())

        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer in self.layers_before:
            if self.use_ckpt:
                x = checkpoint(layer, x, None, padding_mask, False, use_reentrant=True)
            else:
                x = layer(x, None, padding_mask, False)

        x, self.sae_loss = self.sae(x, nonpad_mask, input_mask, method)

        for layer in self.layers_after:
            if self.use_ckpt:
                x = checkpoint(layer, x, None, padding_mask, False, use_reentrant=True)
            else:
                x = layer(x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x
    
    
    
class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x