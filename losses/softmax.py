import torch.nn as nn
import torch
import math


class SoftMaxLossFirstPart(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        #self.weights = nn.Parameter(torch.Tensor(num_features, num_classes))
        self.weights = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.uniform_(self.weights, a=-math.sqrt(3/self.num_features), b=math.sqrt(3/self.num_features))
        nn.init.uniform_(self.bias, a=-math.sqrt(3/self.num_features), b=math.sqrt(3/self.num_features))

    def forward(self, features):
        #affines = features.matmul(self.weights) + self.bias
        affines = features.matmul(self.weights.t()) + self.bias
        logits = affines
        print("softmax")
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
            #targets_one_hot = torch.eye(self.model_classifier.weights.size(1))[targets].long().cuda()
            targets_one_hot = torch.eye(self.model_classifier.weights.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, logits[:len(targets)], torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits[:len(targets)])
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            cls_probabilities = nn.Softmax(dim=1)(logits[:len(targets)])
            ood_probabilities = nn.Softmax(dim=1)(logits[:len(targets)])
            max_logits = logits[:len(targets)].max(dim=1)[0]
            return loss, cls_probabilities, ood_probabilities, max_logits, intra_logits, inter_logits

            """
            if self.training and self.type.split("_")[11].startswith("oe"):
                print("outlier exposure #2")
                ##loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                ##loss += 0.5 * -(logits[len(targets):].mean(1) - torch.logsumexp(logits[len(targets):], dim=1)).mean()
                loss += 0.5 * -(logits[len(targets):2*len(targets)].mean(1) - torch.logsumexp(logits[len(targets):2*len(targets)], dim=1)).mean()
                ##uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
                ##kl_divergence = F.kl_div(F.log_softmax(odd_outputs[:slice_size], dim=1), uniform_dist, reduction='batchmean')
                #uniform_dist = torch.Tensor(len(targets), self.weights.size(0)).fill_((1./self.weights.size(0))).cuda()
                #loss += 0.5 * F.kl_div(F.log_softmax(logits[len(targets):2*len(targets)], dim=1), uniform_dist, reduction='batchmean')
            """
