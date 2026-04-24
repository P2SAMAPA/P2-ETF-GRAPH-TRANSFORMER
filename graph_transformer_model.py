"""
Graph Transformer using TransformerConv from PyG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout))
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)

class GTRunner:
    def __init__(self, in_channels, hidden_channels=64, num_heads=4, num_layers=3, lr=0.001, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphTransformer(in_channels, hidden_channels, num_heads, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_graphs(self, graphs, epochs=80):
        self.model.train()
        for g in graphs:
            g.x = g.x.to(self.device)
            g.edge_index = g.edge_index.to(self.device)
            g.y = g.y.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = 0.0
            for g in graphs:
                pred = self.model(g.x, g.edge_index)
                loss += self.criterion(pred, g.y)
            loss /= len(graphs)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    def predict_latest(self, graphs):
        self.model.eval()
        g = graphs[-1]
        g.x = g.x.to(self.device)
        g.edge_index = g.edge_index.to(self.device)
        with torch.no_grad():
            pred = self.model(g.x, g.edge_index)
        return pred.cpu().numpy()
