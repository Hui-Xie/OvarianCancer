# Analyze the correlation between latent vector and its corresponding response

# dicesFilePath =  "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/predictResult/20191023_153046/patientDice.json"
# latentVectorDir =  "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/latent/latent_20191023_153046"
dicesFilePath =  "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/predictResult/20191025_102445/patientDice.json"
latentVectorDir =  "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/latent/latent_20191025_102445"
patientResponsePath = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

aList = range(0,85,2)  #dice range 0% to 85%, step 2%
diceThresholdList=[x/100 for x in aList]
accuracyThreshold = 0.71  # for each feature
F,H,W = 1536,3,3  #Features, Height, Width of latent vector

import json
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # patient response
    with open(patientResponsePath) as f:
        patientResponse = json.load(f)

    # filter Dice
    with open(dicesFilePath) as f:
        rawPatientDice = json.load(f)
    print(f"latent dir: {latentVectorDir}")
    print(f"There are total {len(rawPatientDice)} patients.")
    print(f"Each patient has a latent vector of size {(F,H,W)}")
    print(f"Each feature alone predicts its response through a least square regression along patients dimension.")
    print(f"accuracy Threshold = {accuracyThreshold}, for each feature")

    validationSamples = []
    response1Rate= []
    averageDiceSamples = []
    minAccuracy  = []
    meanAccuracy = []
    medianAccuracy = []
    maxAccuracy  = []
    numBestFeatures = []  # num of features whose predicted accuracy is greater than specific threshold

    for diceThreshold in diceThresholdList:
        patientDice = rawPatientDice.copy()
        for key in list(patientDice):
            if patientDice[key] < diceThreshold:
                del patientDice[key]
        N = len(patientDice)
        validationSamples.append(N)

        dices = 0.0
        for dice in patientDice.values():
            dices += dice
        averageDiceSamples.append(dices/N)

        # get response vector and assemble latent Vectors
        X = np.empty((F, H, W, N), dtype=np.float)  # latent vectors
        Y01 = np.empty((1,N),dtype=np.int) # response in 0, 1 range
        Yn1 = np.empty((1,N),dtype=np.float) # response in -1, 1 range
        for i, key in enumerate(list(patientDice)):
            Y01[0,i] = patientResponse[key]
            Yn1[0,i] = patientResponse[key]
            if 0 == Yn1[0,i]:
                Yn1[0,i]= -1

            filePath = os.path.join(latentVectorDir,key+".npy")
            V = np.load(filePath)
            assert (F,H,W) == V.shape
            X[:,:,:,i] = V
        response1Rate.append(Y01.sum()/N)

        # normalize latentV along patient dimension
        mean = np.mean(X, axis=3)
        std = np.std(X, axis=3)
        mean = np.reshape(np.repeat(mean, N, axis=2), X.shape)
        std  = np.reshape(np.repeat(std, N, axis=2),X.shape)
        X = (X - mean) / std

        # Analysis  least square loss = (Yn1 - (W0+W1X))^2
        # Y is the expand of Yn1, use -1 and 1 to less square regression
        Y = np.reshape(np.repeat(Yn1, F*H*W,axis=0), X.shape)
        # herer W0 and W1 has shape(F,H, W)
        W1= ((X*Y).sum(axis=3) - X.sum(axis=3)*Y.sum(axis=3)/N)/((X*X).sum(axis=3) - X.sum(axis=3)*X.sum(axis=3)/N)
        W0= (Y.sum(axis=3)-W1*X.sum(axis=3))/N

        W1Ex = np.reshape(np.repeat(W1, N, axis=2), X.shape)
        W0Ex = np.reshape(np.repeat(W0, N, axis=2), X.shape)

        sigmoidX = 1.0/(1.0 + np.exp(-(W0Ex+W1Ex*X)))
        predictX = (sigmoidX >= 0.5).astype(np.int)
        Y = np.reshape(np.repeat(Y01, F*H*W,axis=0), X.shape)
        accuracyX = ((predictX - Y)==0).sum(axis=3)*1.0/N

        minAccuracy.append(accuracyX.min())
        meanAccuracy.append(accuracyX.mean())
        medianAccuracy.append(np.median(accuracyX))
        maxAccuracy.append(accuracyX.max())

        accuracyBig = accuracyX >=accuracyThreshold
        numBestFeaturesTemp  = accuracyBig.sum()
        numBestFeatures.append(numBestFeaturesTemp)

    # print table:
    print(f"dice threshold list:    {diceThresholdList}")
    print(f"validation patients:    {validationSamples}")
    print(f"avgDice of validaiton:  {averageDiceSamples}")
    print(f"Rate of Response 1:     {response1Rate}")
    print(f"minAccuracy:            {minAccuracy}")
    print(f"meanAccuracy:           {meanAccuracy}")
    print(f"medianAccuracy:         {medianAccuracy}")
    print(f"maxAccuracy:            {maxAccuracy}")
    print(f"num of Best Features:   {numBestFeatures}")
    nFeatures = F*H*W
    rateBestFeatures = [x/(nFeatures) for x in numBestFeatures ]
    print(f"rate of Best Features:  {rateBestFeatures}")


    #draw table:
    f = plt.figure(1)
    plt.plot(diceThresholdList, minAccuracy)
    plt.plot(diceThresholdList, meanAccuracy)
    plt.plot(diceThresholdList, medianAccuracy)
    plt.plot(diceThresholdList, maxAccuracy)
    plt.plot(diceThresholdList, response1Rate)
    plt.plot(diceThresholdList, averageDiceSamples)
    plt.gca().set_ylim([0, 1.0])
    plt.legend(('Min', 'Mean', 'median', 'Max', 'resp1Rate', 'avgDice'), loc='upper left')
    plt.title(f"Single-Feature prediciton on different dice thresholds")
    plt.xlabel('Dice Thresholds')
    plt.ylabel('Prediction Accuracy')
    plt.savefig(os.path.join(latentVectorDir, f"SingleFeaturePrediction.png"))
    plt.close()

    f = plt.figure(2)
    plt.plot(diceThresholdList, rateBestFeatures)
    plt.gca().set_ylim([0, 1.0])
    plt.title(f"Rate of Best Features on different dice thresholds")
    plt.xlabel('Dice Thresholds')
    plt.ylabel('Rate of Best Features')
    plt.savefig(os.path.join(latentVectorDir, f"rateBestFeatures.png"))
    plt.close()

if __name__ == "__main__":
    main()



