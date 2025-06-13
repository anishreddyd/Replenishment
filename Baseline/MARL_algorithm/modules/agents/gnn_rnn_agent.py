import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, edge_index):

        """Efficient graph attention without Python loops."""
        Wh = self.fc(h)  # [B, N, F]
        src, dst = edge_index
        src_h = Wh[:, src]  # [B, E, F]
        dst_h = Wh[:, dst]
        e = self.leakyrelu(self.attn_fc(torch.cat([src_h, dst_h], dim=-1))).squeeze(-1)

        dst_expand = dst.unsqueeze(0).expand(e.size(0), -1)
        e_exp = torch.exp(e)
        norm = torch.zeros(e.size(0), Wh.size(1), device=h.device)
        norm.scatter_add_(1, dst_expand, e_exp)
        attn = e_exp / (norm.gather(1, dst_expand) + 1e-6)

        out = torch.zeros_like(Wh)
        src_h_weighted = src_h * attn.unsqueeze(-1)
        out.scatter_add_(1, dst_expand.unsqueeze(-1).expand_as(src_h_weighted), src_h_weighted)
        return F.elu(out)

class GNNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.gat = GraphAttentionLayer(args.hidden_dim, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        self.edge_index = torch.tensor(args.edge_index, dtype=torch.long)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        x = F.relu(self.fc1(inputs), inplace=True)
        edge_index = self.edge_index.to(inputs.device)
        x = self.gat(x, edge_index)
        x = x.view(-1, self.args.hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)
        q = self.fc2(hh)
        return q.view(b, a, -1), hh.view(b, a, -1)

