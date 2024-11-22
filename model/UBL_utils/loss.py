from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss, CosineSimilarity
import torch
def loss_StructureEmbedding_regression(dist, labels):
    criterion_regression = MSELoss()
    labels_regression = labels.T.reshape(-1)
    loss = criterion_regression(dist, labels_regression)
    return loss

def loss_StructureEmbedding_classification(logits, labels):
    criterion_classification = CrossEntropyLoss()
    labels_regression = labels.T.reshape(-1)
    labels_classification = torch.floor((labels_regression+1)*10).to(dtype = torch.long)
    labels_classification[labels_classification < 6] = 0  #0.1-0.3
    labels_classification[(labels_classification >= 6) & (labels_classification<10)] = 1  #0.3-0.5
    labels_classification[(labels_classification >= 10) & (labels_classification<16)] = 2 #0.5-0.75
    labels_classification[labels_classification >= 16] = 3  #0.75-1
    loss = criterion_classification(logits, labels_classification)
    return loss