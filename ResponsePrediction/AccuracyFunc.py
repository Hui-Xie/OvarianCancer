
def computeAccuracy(y, gt):
    '''
    y:  logits before probility
    gt: ground truth
    '''
    y = (y>=0.0).squeeze().int()
    N = gt.shape[0]
    gt = gt.squeeze().int()
    accuracy = ((y - gt) == 0).sum()*1.0 / N
    return accuracy

# accuracy = TPR*P/T + TNR*N/T where T is total number

def computeTNR(y,gt): # True Negative Rate, Specificity
    y = (y >= 0.0).squeeze().int()
    N = gt.shape[0]
    gt = gt.squeeze().int()
    TNR = ((y+gt)==0).sum()*1.0 / (N-gt.sum())
    return TNR

def computeTPR(y, gt): #True Positive Rate, sensitivity
    y = (y >= 0.0).squeeze().int()
    gt = gt.squeeze().int()
    TPR = ((y*gt)==1).sum()*1.0/gt.sum()
    return TPR