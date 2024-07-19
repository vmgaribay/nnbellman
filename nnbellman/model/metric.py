import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def category_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == torch.argmax(target, dim=1)).item()
    return correct / len(target)

def high_adapt_falsepositive(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 2) & (target != 2)).item()

def high_adapt_falsenegative(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 2) & (target == 2)).item()

def low_adapt_falsepositive(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 1) & (target != 1)).item()

def low_adapt_falsenegative(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 1) & (target == 1)).item()

def no_adapt_falsepositive(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 0) & (target != 0)).item()

def no_adapt_falsenegative(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 0) & (target == 0)).item()

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def model_mae(output, target):
    return torch.mean(torch.abs(output - target))

def model_max_ae(output, target):
    return torch.max(torch.abs(output - target))

def model_mse(output, target):
    return torch.mean((output - target)**2)

def consumption_mae(output, target, scale):
    return torch.mean(torch.abs(output[:,1]-target[:,1]))*scale

def consumption_max_ae(output, target, scale):
    return torch.max(torch.abs(output[:,1]-target[:,1]))*scale

def i_a_mae(output, target):
    return torch.mean(torch.abs(output[:,0]-target[:,0]))

def i_a_max_ae(output, target):
    return torch.max(torch.abs(output[:,0]-target[:,0]))

def n_wrong_i_a(output, target, possible_targets):
    return torch.sum(torch.abs(possible_targets[torch.argmin(torch.abs(output[:,0].unsqueeze(-1) - possible_targets), dim=-1)]-target[:,0]) > 0).item()

def n_exceeding_i_a_k(data, output, possible_targets, scale):
    if data.size(1)==4:    
        return torch.sum(data[:,1]-possible_targets[torch.argmin(torch.abs(output[:,0].unsqueeze(-1) - possible_targets), dim=-1)]-scale*output[:,1]< 0).item()
    elif data.size(1)==5:  
        return torch.sum(data[:,1]-data[:,4]-scale*output[:,0]< 0).item()
  
def n_exceeding_k(data, output, scale):    
    return torch.sum(data[:,1]-scale*output[:,0]< 0).item()