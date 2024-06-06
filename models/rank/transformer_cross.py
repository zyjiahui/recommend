# 基于最原始的transformer实现特征交叉，使用nn.Transformer库
import torch
import torch.nn as nn

class FeatureCrossTransformer(nn.Module):
    def __init__(self,num_features,embedded_dim,num_heads,num_layers,hidden_dim,dropout=0.1):
        super(FeatureCrossTransformer,self).__init__()
        self.embedding = nn.Embedding(num_features,embedded_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedded_dim,nhead=num_heads,dim_feedforward=hidden_dim,dropout=dropout)
        self.tansformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc = nn.Linear(embedded_dim,1)

    def forward(self,x):
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.permute(1,0,2)
        transformer_output = self.tansformer_encoder(embedded_x)
        transformer_output = transformer_output.permute(1,0,2)
        pooled_output = torch.mean(transformer_output,dim=1)
        output = torch.sigmoid(self.fc(pooled_output))
        return output






