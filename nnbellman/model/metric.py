import torch


def accuracy(output, target):
    '''Original Template, accuracy of predictions.'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def category_accuracy(output, target):
    '''Ratio of correct adaptation (i_a) predictions'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == torch.argmax(target, dim=1)).item()
    return correct / len(target)

def high_adapt_falsepositive(output, target):
    '''Count of false positives for high adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 2) & (target != 2)).item()

def high_adapt_falsenegative(output, target):
    '''Count of false negatives for high adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 2) & (target == 2)).item()

def low_adapt_falsepositive(output, target):
    '''Count of false positives for low adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 1) & (target != 1)).item()

def low_adapt_falsenegative(output, target):
    '''Count of false negatives for low adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 1) & (target == 1)).item()

def no_adapt_falsepositive(output, target):
    '''Count of false positives for no adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred == 0) & (target != 0)).item()

def no_adapt_falsenegative(output, target):
    '''Count of false negatives for no adaptation'''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        assert pred.shape[0] == len(target)
    return torch.sum((pred != 0) & (target == 0)).item()

def top_k_acc(output, target, k=3):
    '''Original Template, accuracy of top k predictions'''
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def model_mae(output, target):
    '''Mean absolute error of model predictions'''
    return torch.mean(torch.abs(output - target))

def model_max_ae(output, target):
    '''Maximum absolute error of model predictions'''
    return torch.max(torch.abs(output - target))

def model_mse(output, target):
    '''Mean squared error of model predictions'''
    return torch.mean((output - target)**2)

def model_mpe(output, target):
    '''Mean percentage error of model predictions'''
    return torch.mean(torch.abs((output - target) / target)) * 100

def model_max_pe(output, target):    
    '''Maximum percentage error of model predictions'''
    return torch.max(torch.abs((output - target) / target)) * 100

def consumption_mae(output, target, cons_scale):
    '''Mean absolute error of consumption predictions'''
    return torch.mean(torch.abs(output[:,1]-target[:,1]))*cons_scale

def consumption_max_ae(output, target, cons_scale):
    '''Maximum absolute error of consumption predictions'''
    return torch.max(torch.abs(output[:,1]-target[:,1]))*cons_scale

def consumption_mpe(output, target, cons_scale):    
    '''Mean percentage error of consumption predictions'''
    if output.size(1)==2:
        return torch.mean(torch.abs((output[:,1]-target[:,1])/target[:,1]))*cons_scale*100
    elif output.size(1)==1:
        return torch.mean(torch.abs((output[:,0]-target[:,0])/target[:,0]))*cons_scale*100
    else:
        print("ERROR: Output size not as expected")

def consumption_max_pe(output, target, cons_scale):    
    '''Maximum percentage error of consumption predictions'''
    if output.size(1)==2:
        return torch.max(torch.abs((output[:,1]-target[:,1])/target[:,1]))*cons_scale*100
    elif output.size(1)==1:
        return torch.max(torch.abs((output[:,0]-target[:,0])/target[:,0]))*cons_scale*100

def i_a_mae(output, target, i_a_scale):
    '''Mean absolute error of adaptation predictions'''
    return torch.mean(torch.abs(output[:,0]-target[:,0]))*i_a_scale

def i_a_max_ae(output, target,i_a_scale):
    '''Maximum absolute error of adaptation predictions'''
    return torch.max(torch.abs(output[:,0]-target[:,0]))*i_a_scale

def i_a_mpe(output, target, i_a_scale):
    '''Mean percentage error of adaptation predictions'''
    return torch.mean(torch.abs((output[:,0]-target[:,0])/target[:,0]))*i_a_scale*100

def i_a_maxpe(output, target, i_a_scale):
    '''Maximum percentage error of adaptation predictions'''
    return torch.max(torch.abs((output[:,0]-target[:,0])/target[:,0]))*i_a_scale*100

def n_wrong_i_a(output, target, possible_targets, i_a_scale):
    '''Count of incorrect adaptation predictions'''
    return torch.sum(torch.abs(possible_targets[torch.argmin(torch.abs(output[:,0].unsqueeze(-1)*i_a_scale - possible_targets), dim=-1)]-possible_targets[torch.argmin(torch.abs(target[:,0].unsqueeze(-1)*i_a_scale - possible_targets), dim=-1)]) > 0).item()

def n_exceeding_i_a_k(data, output, possible_targets, cons_scale, i_a_scale, input_scale):
    '''
    Count of consumption predictions exceeding capital minus adaptation investment 
    (not relevant for new past shock Bellman)
    '''
    if 'k' not in input_scale.keys():
        scaled_k = data[:,1]
    elif input_scale['k']['dist'] in ["unif","uniform"]:
        scaled_k = data[:,1]*(input_scale['k']['params'][1]-input_scale['k']['params'][0])+input_scale['k']['params'][0]
    elif input_scale['k']['dist'] in ["norm","normal"]:
        scaled_k = data[:,1]*input_scale['k']['params'][1]+input_scale['k']['params'][0]
    if data.size(1)==4:    
        return torch.sum(scaled_k-possible_targets[torch.argmin(torch.abs(output[:,0].unsqueeze(-1)*i_a_scale - possible_targets), dim=-1)]-cons_scale*output[:,1]< 0).item()
    elif data.size(1)==5:  
        return torch.sum(scaled_k-data[:,4]-cons_scale*output[:,0]< 0).item()

  
def n_exceeding_k(data, output, cons_scale, input_scale): 
    '''
    Count of consumption predictions exceeding capital
    (not relevant for new past shock Bellman)
    ''' 
    if 'k' not in input_scale.keys():
        scaled_k = data[:,1]
    elif input_scale['k']['dist'] in ["unif","uniform"]:
        scaled_k = data[:,1]*(input_scale['k']['params'][1]-input_scale['k']['params'][0])+input_scale['k']['params'][0]
    elif input_scale['k']['dist'] in ["norm","normal"]:
        scaled_k = data[:,1]*input_scale['k']['params'][1]+input_scale['k']['params'][0]
    return torch.sum(scaled_k-cons_scale*output[:,0]< 0).item()