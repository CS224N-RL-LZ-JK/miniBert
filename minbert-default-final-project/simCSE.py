import torch
import torch.nn.functional as F

class SimCSEModel(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(SimCSEModel, self).__init__()
        self.encoder = pretrained_model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use the [CLS] token's embeddings
        pooled_output = outputs.last_hidden_state[:, 0]
        return pooled_output

    @staticmethod
    def contrastive_loss(features, temperature=0.05):
        """
        Compute the contrastive loss given a batch of feature vectors.
        """
        # Normalize features to prevent large values of cosine similarity, improve stability
        features = F.normalize(features, p=2, dim=1)
        cos_sim = torch.matmul(features, features.T)
        cos_sim = cos_sim / temperature
        
        labels = torch.arange(features.size(0), device=features.device)
        labels = labels.long()
        # Ensure labels are correct for contrastive learning; might need adjustment based on batching/pairing strategy
        
        loss_fct = torch.nn.CrossEntropyLoss()
        # Diagonal elements are not compared against themselves
        labels = torch.arange(cos_sim.size(0), device=cos_sim.device)
        loss = loss_fct(cos_sim, labels)
        return loss
