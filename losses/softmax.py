import torch.nn as nn
import torch
import math


class SoftMaxLossFirstPart(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.Tensor(num_features, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.uniform_(self.weights, a=-math.sqrt(3/self.num_features), b=math.sqrt(3/self.num_features))
        nn.init.uniform_(self.bias, a=-math.sqrt(3/self.num_features), b=math.sqrt(3/self.num_features))

    def forward(self, features):
        affines = features.matmul(self.weights) + self.bias
        logits = affines
        #print("softmax")
        return logits


class SoftMaxLossSecondPart(nn.Module):
    def __init__(self, model_classifier):
        super(SoftMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, debug=False):
        loss = self.loss(logits, targets)
        if not debug:
            return loss
        else:
            print("softmax")
            targets_one_hot = torch.eye(self.model_classifier.weights.size(1))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            cls_probabilities = nn.Softmax(dim=1)(logits)
            ood_probabilities = nn.Softmax(dim=1)(logits)
            max_logits = logits.max(dim=1)[0]
            return loss, cls_probabilities, ood_probabilities, max_logits, intra_logits, inter_logits