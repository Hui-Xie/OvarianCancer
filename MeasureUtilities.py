# Measurement Utilities

import numpy as np

def getAccuracy(predicts, labels):
    """
    :param predicts:  nump array for n predicts
    :param labels:    nump array for n labels.
    :return:
    """
    if labels is None:
        return 0
    nSame = np.equal(predicts,labels).sum()
    return nSame/labels.shape[0]



def getDiceSumList(segmentations, labels, K):
    """
    :param segmentations: with N samples
    :param labels: ground truth with N samples
    :param K : number of classification
    :return: (diceSumList,diceCountList)
            diceSumList: whose element 0 indicates total dice sum over N samples, element 1 indicate label1 dice sum over N samples, etc
            diceCountList: indicate effective dice count
    :Notes: when labels <0, its dice will not be computed.
            support 2D and 3D array.
    """
    N = segmentations.shape[0]  # sample number
    diceSumList = [0 for _ in range(K)]
    diceCountList = [0 for _ in range(K)]
    for i in range(N):
        (dice,count) = getDice((segmentations[i] > 0) , (labels[i] > 0) )
        diceSumList[0] += dice
        diceCountList[0] += count
        for j in range(1, K):
            (dice, count) = getDice((segmentations[i]==j), (labels[i]==j))
            diceSumList[j] += dice
            diceCountList[j] += count

    return diceSumList, diceCountList


def getDice(segmentation, label):
    """

    :param segmentation:  0-1 elements array
    :param label:  0-1 elements array
    :return: (dice, count) count=1 indicates it is an effective dice, count=0 indicates there is no nonzero elements in label.
    :Notes: support 2D and 3D array,
            value <0 will be ignored.
    """
    seg1 = segmentation >0
    label1 = label >0
    nA = np.count_nonzero(seg1)
    nB = np.count_nonzero(label1 )
    C = seg1 * label1
    nC = np.count_nonzero(C)
    if 0 == nB:  # the dice was calculated over the slice where a ground truth was available.
        return 0, 0
    else:
        return nC*2.0/(nA+nB), 1


def getTPR(predict, label):  # sensitivity, recall, hit rate, or true positive rate (TPR)
    """
    :param predict:
    :param label:
    :return: A tuple
    :Notes: support 2D and 3D array,
            value <0 will be ignored.
    """
    if predict is None:
        return 0, 0
    seg1 = predict > 0
    label1 = label >0
    nB = np.count_nonzero(label1)
    C = seg1 * label1
    nC = np.count_nonzero(C)
    if 0 == nB:
        return 0, 0
    else:
        return nC / nB, 1


def getTNR(predict, label):  # specificity, selectivity or true negative rate (TNR)
    """
    :param predict:
    :param label:
    :return: A tuple
    :Notes: support 2D and 3D array,
            value <0 will be ignored.
    """
    if predict is None:
        return 0, 0
    seg1 = predict == 0
    label1 = label == 0
    nB = np.count_nonzero(label1)
    C = seg1 * label1
    nC = np.count_nonzero(C)
    if 0 == nB:
        return 0, 0
    else:
        return nC / nB, 1



def getTPRSumList(predicts, labels, K):
    """
    :param predicts: with N samples
    :param labels: ground truth with N samples
    :param K:   classification number
    :return: (TPRSumList,TPRCountList)
            TPRSumList: whose element 0 indicates total TPR sum over N samples, element 1 indicate label1 TPR sum over N samples, etc
            TPRCountList: indicate effective TPR count
    :Notes: support 2D and 3D array,
            value <0 will be ignored.
    """
    N = predicts.shape[0]  # sample number

    TPRSumList = [0 for _ in range(K)]
    TPRCountList = [0 for _ in range(K)]
    for i in range(N):
        (TPR, count) = getTPR((predicts[i] > 0), (labels[i] > 0))
        TPRSumList[0] += TPR
        TPRCountList[0] += count
        for j in range(1, K):
            (TPR, count) = getTPR((predicts[i] == j), (labels[i] == j))
            TPRSumList[j] += TPR
            TPRCountList[j] += count

    return TPRSumList, TPRCountList
