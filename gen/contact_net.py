''






import torch
import torch.nn as nn


class ContactNet(nn.Module):
    def __init__(self, n_joints=22, n_feet=4, d_model=256, n_layers=4, n_heads=8, ff=1024, max_len=128, dropout=0.0):
        super().__init__()
        self.n_joints = n_joints; self.n_feet = n_feet
        self.in_dim = n_joints * 4 + n_joints * 3
        self.inp = nn.Linear(self.in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_feet)

    def forward(self, q, om_n):
        ''
        B, N = q.shape[:2]
        x = torch.cat([q.reshape(B, N, -1), om_n.reshape(B, N, -1)], dim=-1)
        h = self.inp(x) + self.pos[:, :N]
        h = self.encoder(h)
        return self.head(h)

    @torch.no_grad()
    def predict(self, q, om_n, thresh=0.5):
        ''
        return (torch.sigmoid(self.forward(q, om_n)) > thresh).float()
