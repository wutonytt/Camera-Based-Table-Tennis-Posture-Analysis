import os, json
import pandas as pd
import numpy as np

def loadTrainData(dirPath, numOfSet):

    train = pd.DataFrame()

    for d in range(1, numOfSet + 1):
        setDir = os.path.join(dirPath, 'train_' + str(d))
        prefix = os.path.join(setDir + '/' + str(d) + '_')
        numOfFiles = len([file_ for file_ in os.listdir(setDir) if os.path.isfile(os.path.join(setDir, file_))])
        
        for file_num in range(numOfFiles):
            f = open(prefix + str(file_num).zfill(12) + '_keypoints.json')
            j = json.load(f)

            leftData = [[]]
            rightData = [[]]

            if (j['people'] != []):
                for i in j['people']:
                    counterr = 0
                    counterl = 0
                    for x in i['pose_keypoints_2d'][::3]:
                        if (x >= 700):
                            counterr += 1
                        if (x <= 200):
                            counterl += 1
                    if (counterr > 15):
                        rightData[0] += i['pose_keypoints_2d']
                    elif (counterl > 15):
                        leftData[0] += i['pose_keypoints_2d']

            columns =  ['train_num', 'file_num']
            for i in range(25):
                X = 'X' + str(i)
                Y = 'Y' + str(i)
                P = 'P' + str(i)
                columns += [X, Y, P]

            if (len(leftData[0]) == 75):
                leftData = [d, file_num] + leftData[0]
                dfl = pd.DataFrame([leftData], columns = columns)
                dfl['left/right'] = 0
                train = train.append(dfl, ignore_index=True)

            if (len(rightData[0]) == 75):
                rightData = [d, file_num] + rightData[0]
                dfr = pd.DataFrame([rightData], columns = columns)
                dfr['left/right'] = 1
                train = train.append(dfr, ignore_index=True)

    train = train.drop(list(train.filter(like='P', axis=1)), axis = 1)

    return train

def loadTestData(dirPath, frontName):

    test = pd.DataFrame()

    path = os.path.join(dirPath, frontName)
    numOfFiles = len([name for name in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, name))])
    
    for file_num in range(numOfFiles):
        f = open(path + str(file_num).zfill(12) + '_keypoints.json')
        j = json.load(f)

        leftData = [[]]
        rightData = [[]]

        if (j['people'] != []):
            for i in j['people']:
                counterr = 0
                counterl = 0
                for k in i['pose_keypoints_2d'][::3]:
                    if (k >= 700):
                        counterr += 1
                    if (k <= 200):
                        counterl += 1
                if (counterr > 15):
                    rightData[0] += i['pose_keypoints_2d']
                elif (counterl > 15):
                    leftData[0] += i['pose_keypoints_2d']

        columns =  ['file_num']
            for i in range(25):
                X = 'X' + str(i)
                Y = 'Y' + str(i)
                P = 'P' + str(i)
                columns += [X, Y, P]

        if (len(leftData[0]) == 75):
            leftData = [file_num] + leftData[0]
            dfl = pd.DataFrame ([leftData], columns = columns)
            dfl['left/right'] = 0
            test = test.append(dfl, ignore_index=True)

        if (len(rightData[0]) == 75):
            rightData = [file_num] + rightData[0]
            dfr = pd.DataFrame ([rightData], columns = columns)
            dfr['left/right'] = 1
            test = test.append(dfr, ignore_index=True)

    test = test.drop(list(test.filter(like='P', axis=1)), axis = 1)
    return test


def addTrainLabel(train, labelFile):
    labeldf = pd.read_csv(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/' + labelFile)
    for index, row in labeldf.iterrows():
        row['file_num'] = 3 * round(row['file_num']/3)
    train = pd.merge(train, labeldf, on=['train_num', 'file_num','left/right'], how = 'inner')
    return train


def addTestLabel(test, labelFile, test_num):
    labeldf = pd.read_csv(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/' + labelFile)
    for index, row in labeldf.iterrows():
        row['file_num'] = 3 * round(row['file_num']/3)
    labeldf = labeldf[labeldf['train_num'] == test_num]
    test = pd.merge(test, labeldf, on=['file_num','left/right'], how = 'inner')
    return test


def dataAugmentation(train):
    tmp0 = train.copy()
    fore = tmp0[tmp0['fore/back'] == 1]
    back = tmp0[tmp0['fore/back'] == 0]

    # fore
    for i in range(-5,6):
        tmp = fore.copy()
        if i == 0 :
            continue
        cols = train.iloc[:,2:-1].columns
        tmp[cols] += i
        train = train.append(tmp, ignore_index=True)

    # back
    for i in range(-7,8):
        tmp = back.copy()
        if i == 0 :
            continue
        cols = train.iloc[:,2:-1].columns
        tmp[cols] += i
        train = train.append(tmp, ignore_index=True)
    
    return train