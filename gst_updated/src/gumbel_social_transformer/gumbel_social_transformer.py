import torch
import torch.nn as nn
from gst_updated.src.gumbel_social_transformer.utils import _get_clones
from gst_updated.src.gumbel_social_transformer.group_generator import GroupGenerator
from gst_updated.src.gumbel_social_transformer.edge_selector_no_ghost import EdgeSelector
from gst_updated.src.gumbel_social_transformer.node_encoder_layer_no_ghost import NodeEncoderLayer
from torch.nn.functional import softmax
from gst_updated.src.gumbel_social_transformer.mha import VanillaMultiheadAttention
from gst_updated.src.gumbel_social_transformer.utils import _get_activation_fn, gumbel_softmax

class GumbelSocialTransformer(nn.Module):
    def __init__(self, d_motion, d_model, nhead_nodes, nhead_edges, nlayer, dim_feedforward=512, dim_hidden=32, \
        dropout=0.1, activation="relu", attn_mech="vanilla", ghost=True):
        super(GumbelSocialTransformer, self).__init__()
        self.ghost = ghost
        # Common components
        self.edge_selector = EdgeSelector(d_model, nhead=nhead_edges, dropout=dropout, activation=activation)
        self.node_embedding = nn.Linear(d_motion, d_model)
        node_encoder_layer = NodeEncoderLayer(d_model, nhead_nodes, dim_feedforward, dropout, activation, attn_mech)
        self.node_encoder_layers = _get_clones(node_encoder_layer, nlayer)
        # GroupGenerator initialization with default parameters
        self.group_gen = GroupGenerator()

    def forward(self, x, A, attn_mask, tau=1., hard=False, device='cuda:0'):
        # Preprocess x for GroupGenerator
        # Placeholder: Modify according to your specific input needs for GroupGenerator
        v_abs = self.node_embedding(x)  # Assuming v_abs is derived after node embedding
        _, indices = self.group_gen(v_abs, v_abs)  # Use v_abs for both parameters as a placeholder

        # Actual EdgeSelector and NodeEncoderLayer process
        if self.ghost:
            # Placeholder: Modify according to your specific input needs for EdgeSelector
            edge_multinomial, sampled_edges = self.edge_selector(v_abs, A, attn_mask, tau, hard, device)
        else:
            bsz, nnode = attn_mask.shape[0], attn_mask.shape[1]
            sampled_edges = torch.ones(bsz, nnode, 1, nnode).to(device) * attn_mask.unsqueeze(2)
            edge_multinomial = torch.ones(bsz, nnode, 1, nnode).to(device) * attn_mask.unsqueeze(2)  # fake edge_multinomial

        attn_weights_list = []
        for i in range(self.nlayer):
            v_abs, attn_weights_layer = self.node_encoder_layers[i](v_abs, sampled_edges, attn_mask, device)
            attn_weights_list.append(attn_weights_layer)
        attn_weights = torch.stack(attn_weights_list, dim=0) # (nlayer, bsz, nhead, nnode, nnode)
        
        return v_abs, sampled_edges, edge_multinomial, attn_weights

# Example instantiation and forward call placeholder
# model = GumbelSocialTransformer(d_motion=32, d_model=512, nhead_nodes=8, nhead_edges=4, nlayer=6, dropout=0.1, activation="relu", attn_mech="vanilla", ghost=False)
# x = torch.rand(10, 20, 32)  # (bsz, nnode, d_motion)
# A = torch.rand(10, 20, 20, 32)  # (bsz, nnode, nnode, d_motion)
# attn_mask = torch.ones(10, 20, 20)  # (bsz, nnode, nnode)
# tau = 1.0
# hard = False
# device = 'cuda:0'
# outputs = model(x, A, attn_mask, tau, hard, device)

