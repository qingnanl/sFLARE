# src/yourpkg/models.py
import torch
import torch.nn as nn

class BipartiteEmbeddingModel(nn.Module):
    def __init__(self, num_proteins, num_genes, embedding_dim, prior_cell_embedding):
        super().__init__()
        self.protein_emb = nn.Embedding(num_proteins, embedding_dim)
        self.gene_emb = nn.Embedding(num_genes, embedding_dim)
        with torch.no_grad():
            self.protein_emb.weight.copy_(prior_cell_embedding)
            nn.init.xavier_uniform_(self.gene_emb.weight)

    def forward(self, edges_protein, edges_gene):
        p_vec = self.protein_emb(edges_protein)
        g_vec = self.gene_emb(edges_gene)
        return p_vec, g_vec
