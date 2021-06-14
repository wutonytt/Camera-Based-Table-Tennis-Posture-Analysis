import os, json
import pandas as pd
import numpy as np

def loadTrainData(dirPath, numOfSet):

    train = pd.DataFrame()

    for d in range(1, numOfSet + 1):
        # print(dirPath)
        dirpath = os.path.join(dirPath, 'train_' + str(d))
        path = os.path.join(dirPath, 'train_' + str(d) + '/' + str(d) + '_')
        numOfFiles = len([name for name in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, name))]) - 3
        # print(numofFiles)
        for file_num in range(0, numOfFiles, 3):
            file1 = open(path + str(file_num).zfill(12) + '_keypoints.json')
            file2 = open(path + str(file_num+1).zfill(12) + '_keypoints.json')
            file3 = open(path + str(file_num+2).zfill(12) + '_keypoints.json')

            j1 = json.load(file1)
            j2 = json.load(file2)
            j3 = json.load(file3)

            leftData = [[]]
            rightData = [[]]

            for j in [j1, j2, j3]:
                if (j['people'] != [] and j['people'] != [] and j['people'] != []):
                    for i in j['people']:
                        counterr = 0
                        for k in i['pose_keypoints_2d'][::3]:
                            if (k >= 700):
                                counterr += 1
                        if (counterr > 15):
                            rightData[0] += i['pose_keypoints_2d']

                        counterl = 0
                        for k in i['pose_keypoints_2d'][::3]:
                            if (k <= 200):
                                counterl += 1
                        if (counterl > 15):
                            leftData[0] += i['pose_keypoints_2d']

            if (len(leftData[0]) == 225):
                leftData = [d] + [file_num] + leftData[0]
                dfl = pd.DataFrame ([leftData], columns = ['train_num', 'file_num', 'First_X0', 'First_Y0', 'First_P0','First_X1', 'First_Y1', 'First_P1','First_X2', 'First_Y2', 'First_P2','First_X3', 'First_Y3', 'First_P3','First_X4', 'First_Y4', 'First_P4','First_X5', 'First_Y5', 'First_P5','First_X6', 'First_Y6', 'First_P6','First_X7', 'First_Y7', 'First_P7','First_X8', 'First_Y8', 'First_P8','First_X9', 'First_Y9', 'First_P9','First_X10', 'First_Y10', 'First_P10','First_X11', 'First_Y11', 'First_P11','First_X12', 'First_Y12', 'First_P12','First_X13', 'First_Y13', 'First_P13','First_X14', 'First_Y14', 'First_P14','First_X15', 'First_Y15', 'First_P15','First_X16', 'First_Y16', 'First_P16','First_X17', 'First_Y17', 'First_P17','First_X18', 'First_Y18', 'First_P18','First_X19', 'First_Y19', 'First_P19','First_X20', 'First_Y20', 'First_P10','First_X21', 'First_Y21', 'First_P21','First_X22', 'First_Y22', 'First_P22','First_X23', 'First_Y23', 'First_P23','First_X24', 'First_Y24', 'First_P24', 'Second_X0', 'Second_Y0', 'Second_P0','Second_X1', 'Second_Y1', 'Second_P1','Second_X2', 'Second_Y2', 'Second_P2','Second_X3', 'Second_Y3', 'Second_P3','Second_X4', 'Second_Y4', 'Second_P4','Second_X5', 'Second_Y5', 'Second_P5','Second_X6', 'Second_Y6', 'Second_P6','Second_X7', 'Second_Y7', 'Second_P7','Second_X8', 'Second_Y8', 'Second_P8','Second_X9', 'Second_Y9', 'Second_P9','Second_X10', 'Second_Y10', 'Second_P10','Second_X11', 'Second_Y11', 'Second_P11','Second_X12', 'Second_Y12', 'Second_P12','Second_X13', 'Second_Y13', 'Second_P13','Second_X14', 'Second_Y14', 'Second_P14','Second_X15', 'Second_Y15', 'Second_P15','Second_X16', 'Second_Y16', 'Second_P16','Second_X17', 'Second_Y17', 'Second_P17','Second_X18', 'Second_Y18', 'Second_P18','Second_X19', 'Second_Y19', 'Second_P19','Second_X20', 'Second_Y20', 'Second_P10','Second_X21', 'Second_Y21', 'Second_P21','Second_X22', 'Second_Y22', 'Second_P22','Second_X23', 'Second_Y23', 'Second_P23','Second_X24', 'Second_Y24', 'Second_P24', 'Third_X0', 'Third_Y0', 'Third_P0','Third_X1', 'Third_Y1', 'Third_P1','Third_X2', 'Third_Y2', 'Third_P2','Third_X3', 'Third_Y3', 'Third_P3','Third_X4', 'Third_Y4', 'Third_P4','Third_X5', 'Third_Y5', 'Third_P5','Third_X6', 'Third_Y6', 'Third_P6','Third_X7', 'Third_Y7', 'Third_P7','Third_X8', 'Third_Y8', 'Third_P8','Third_X9', 'Third_Y9', 'Third_P9','Third_X10', 'Third_Y10', 'Third_P10','Third_X11', 'Third_Y11', 'Third_P11','Third_X12', 'Third_Y12', 'Third_P12','Third_X13', 'Third_Y13', 'Third_P13','Third_X14', 'Third_Y14', 'Third_P14','Third_X15', 'Third_Y15', 'Third_P15','Third_X16', 'Third_Y16', 'Third_P16','Third_X17', 'Third_Y17', 'Third_P17','Third_X18', 'Third_Y18', 'Third_P18','Third_X19', 'Third_Y19', 'Third_P19','Third_X20', 'Third_Y20', 'Third_P10','Third_X21', 'Third_Y21', 'Third_P21','Third_X22', 'Third_Y22', 'Third_P22','Third_X23', 'Third_Y23', 'Third_P23','Third_X24', 'Third_Y24', 'Third_P24'])
                dfl['left/right'] = 0
                train = train.append(dfl, ignore_index=True)

            if (len(rightData[0]) == 225):
                rightData = [d] + [file_num] + rightData[0]
                dfr = pd.DataFrame ([rightData], columns = ['train_num', 'file_num', 'First_X0', 'First_Y0', 'First_P0','First_X1', 'First_Y1', 'First_P1','First_X2', 'First_Y2', 'First_P2','First_X3', 'First_Y3', 'First_P3','First_X4', 'First_Y4', 'First_P4','First_X5', 'First_Y5', 'First_P5','First_X6', 'First_Y6', 'First_P6','First_X7', 'First_Y7', 'First_P7','First_X8', 'First_Y8', 'First_P8','First_X9', 'First_Y9', 'First_P9','First_X10', 'First_Y10', 'First_P10','First_X11', 'First_Y11', 'First_P11','First_X12', 'First_Y12', 'First_P12','First_X13', 'First_Y13', 'First_P13','First_X14', 'First_Y14', 'First_P14','First_X15', 'First_Y15', 'First_P15','First_X16', 'First_Y16', 'First_P16','First_X17', 'First_Y17', 'First_P17','First_X18', 'First_Y18', 'First_P18','First_X19', 'First_Y19', 'First_P19','First_X20', 'First_Y20', 'First_P10','First_X21', 'First_Y21', 'First_P21','First_X22', 'First_Y22', 'First_P22','First_X23', 'First_Y23', 'First_P23','First_X24', 'First_Y24', 'First_P24', 'Second_X0', 'Second_Y0', 'Second_P0','Second_X1', 'Second_Y1', 'Second_P1','Second_X2', 'Second_Y2', 'Second_P2','Second_X3', 'Second_Y3', 'Second_P3','Second_X4', 'Second_Y4', 'Second_P4','Second_X5', 'Second_Y5', 'Second_P5','Second_X6', 'Second_Y6', 'Second_P6','Second_X7', 'Second_Y7', 'Second_P7','Second_X8', 'Second_Y8', 'Second_P8','Second_X9', 'Second_Y9', 'Second_P9','Second_X10', 'Second_Y10', 'Second_P10','Second_X11', 'Second_Y11', 'Second_P11','Second_X12', 'Second_Y12', 'Second_P12','Second_X13', 'Second_Y13', 'Second_P13','Second_X14', 'Second_Y14', 'Second_P14','Second_X15', 'Second_Y15', 'Second_P15','Second_X16', 'Second_Y16', 'Second_P16','Second_X17', 'Second_Y17', 'Second_P17','Second_X18', 'Second_Y18', 'Second_P18','Second_X19', 'Second_Y19', 'Second_P19','Second_X20', 'Second_Y20', 'Second_P10','Second_X21', 'Second_Y21', 'Second_P21','Second_X22', 'Second_Y22', 'Second_P22','Second_X23', 'Second_Y23', 'Second_P23','Second_X24', 'Second_Y24', 'Second_P24', 'Third_X0', 'Third_Y0', 'Third_P0','Third_X1', 'Third_Y1', 'Third_P1','Third_X2', 'Third_Y2', 'Third_P2','Third_X3', 'Third_Y3', 'Third_P3','Third_X4', 'Third_Y4', 'Third_P4','Third_X5', 'Third_Y5', 'Third_P5','Third_X6', 'Third_Y6', 'Third_P6','Third_X7', 'Third_Y7', 'Third_P7','Third_X8', 'Third_Y8', 'Third_P8','Third_X9', 'Third_Y9', 'Third_P9','Third_X10', 'Third_Y10', 'Third_P10','Third_X11', 'Third_Y11', 'Third_P11','Third_X12', 'Third_Y12', 'Third_P12','Third_X13', 'Third_Y13', 'Third_P13','Third_X14', 'Third_Y14', 'Third_P14','Third_X15', 'Third_Y15', 'Third_P15','Third_X16', 'Third_Y16', 'Third_P16','Third_X17', 'Third_Y17', 'Third_P17','Third_X18', 'Third_Y18', 'Third_P18','Third_X19', 'Third_Y19', 'Third_P19','Third_X20', 'Third_Y20', 'Third_P10','Third_X21', 'Third_Y21', 'Third_P21','Third_X22', 'Third_Y22', 'Third_P22','Third_X23', 'Third_Y23', 'Third_P23','Third_X24', 'Third_Y24', 'Third_P24'])
                dfr['left/right'] = 1
                train = train.append(dfr, ignore_index=True)

    train = train.drop(list(train.filter(like='P', axis=1)), axis = 1)
    return train

def loadTestData(dirPath, frontName):

    test = pd.DataFrame()

    path = os.path.join(dirPath, frontName)
    numOfFiles = len([name for name in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, name))]) - 3
    for file_num in range(0, numOfFiles, 3):
        file1 = open(path + str(file_num).zfill(12) + '_keypoints.json')
        file2 = open(path + str(file_num+1).zfill(12) + '_keypoints.json')
        file3 = open(path + str(file_num+2).zfill(12) + '_keypoints.json')

        j1 = json.load(file1)
        j2 = json.load(file2)
        j3 = json.load(file3)

        leftData = [[]]
        rightData = [[]]

        for j in [j1, j2, j3]:
            if (j['people'] != [] and j['people'] != [] and j['people'] != []):
                for i in j['people']:
                    counterr = 0
                    for k in i['pose_keypoints_2d'][::3]:
                        if (k >= 700):
                            counterr += 1
                    if (counterr > 15):
                        rightData[0] += i['pose_keypoints_2d']

                    counterl = 0
                    for k in i['pose_keypoints_2d'][::3]:
                        if (k <= 200):
                            counterl += 1
                    if (counterl > 15):
                        leftData[0] += i['pose_keypoints_2d']

        if (len(leftData[0]) == 225):
            leftData = [file_num] + leftData[0]
            dfl = pd.DataFrame ([leftData], columns = ['file_num', 'First_X0', 'First_Y0', 'First_P0','First_X1', 'First_Y1', 'First_P1','First_X2', 'First_Y2', 'First_P2','First_X3', 'First_Y3', 'First_P3','First_X4', 'First_Y4', 'First_P4','First_X5', 'First_Y5', 'First_P5','First_X6', 'First_Y6', 'First_P6','First_X7', 'First_Y7', 'First_P7','First_X8', 'First_Y8', 'First_P8','First_X9', 'First_Y9', 'First_P9','First_X10', 'First_Y10', 'First_P10','First_X11', 'First_Y11', 'First_P11','First_X12', 'First_Y12', 'First_P12','First_X13', 'First_Y13', 'First_P13','First_X14', 'First_Y14', 'First_P14','First_X15', 'First_Y15', 'First_P15','First_X16', 'First_Y16', 'First_P16','First_X17', 'First_Y17', 'First_P17','First_X18', 'First_Y18', 'First_P18','First_X19', 'First_Y19', 'First_P19','First_X20', 'First_Y20', 'First_P10','First_X21', 'First_Y21', 'First_P21','First_X22', 'First_Y22', 'First_P22','First_X23', 'First_Y23', 'First_P23','First_X24', 'First_Y24', 'First_P24', 'Second_X0', 'Second_Y0', 'Second_P0','Second_X1', 'Second_Y1', 'Second_P1','Second_X2', 'Second_Y2', 'Second_P2','Second_X3', 'Second_Y3', 'Second_P3','Second_X4', 'Second_Y4', 'Second_P4','Second_X5', 'Second_Y5', 'Second_P5','Second_X6', 'Second_Y6', 'Second_P6','Second_X7', 'Second_Y7', 'Second_P7','Second_X8', 'Second_Y8', 'Second_P8','Second_X9', 'Second_Y9', 'Second_P9','Second_X10', 'Second_Y10', 'Second_P10','Second_X11', 'Second_Y11', 'Second_P11','Second_X12', 'Second_Y12', 'Second_P12','Second_X13', 'Second_Y13', 'Second_P13','Second_X14', 'Second_Y14', 'Second_P14','Second_X15', 'Second_Y15', 'Second_P15','Second_X16', 'Second_Y16', 'Second_P16','Second_X17', 'Second_Y17', 'Second_P17','Second_X18', 'Second_Y18', 'Second_P18','Second_X19', 'Second_Y19', 'Second_P19','Second_X20', 'Second_Y20', 'Second_P10','Second_X21', 'Second_Y21', 'Second_P21','Second_X22', 'Second_Y22', 'Second_P22','Second_X23', 'Second_Y23', 'Second_P23','Second_X24', 'Second_Y24', 'Second_P24', 'Third_X0', 'Third_Y0', 'Third_P0','Third_X1', 'Third_Y1', 'Third_P1','Third_X2', 'Third_Y2', 'Third_P2','Third_X3', 'Third_Y3', 'Third_P3','Third_X4', 'Third_Y4', 'Third_P4','Third_X5', 'Third_Y5', 'Third_P5','Third_X6', 'Third_Y6', 'Third_P6','Third_X7', 'Third_Y7', 'Third_P7','Third_X8', 'Third_Y8', 'Third_P8','Third_X9', 'Third_Y9', 'Third_P9','Third_X10', 'Third_Y10', 'Third_P10','Third_X11', 'Third_Y11', 'Third_P11','Third_X12', 'Third_Y12', 'Third_P12','Third_X13', 'Third_Y13', 'Third_P13','Third_X14', 'Third_Y14', 'Third_P14','Third_X15', 'Third_Y15', 'Third_P15','Third_X16', 'Third_Y16', 'Third_P16','Third_X17', 'Third_Y17', 'Third_P17','Third_X18', 'Third_Y18', 'Third_P18','Third_X19', 'Third_Y19', 'Third_P19','Third_X20', 'Third_Y20', 'Third_P10','Third_X21', 'Third_Y21', 'Third_P21','Third_X22', 'Third_Y22', 'Third_P22','Third_X23', 'Third_Y23', 'Third_P23','Third_X24', 'Third_Y24', 'Third_P24'])
            dfl['left/right'] = 0
            test = test.append(dfl, ignore_index=True)

        if (len(rightData[0]) == 225):
            rightData = [file_num] + rightData[0]
            dfr = pd.DataFrame ([rightData], columns = ['file_num', 'First_X0', 'First_Y0', 'First_P0','First_X1', 'First_Y1', 'First_P1','First_X2', 'First_Y2', 'First_P2','First_X3', 'First_Y3', 'First_P3','First_X4', 'First_Y4', 'First_P4','First_X5', 'First_Y5', 'First_P5','First_X6', 'First_Y6', 'First_P6','First_X7', 'First_Y7', 'First_P7','First_X8', 'First_Y8', 'First_P8','First_X9', 'First_Y9', 'First_P9','First_X10', 'First_Y10', 'First_P10','First_X11', 'First_Y11', 'First_P11','First_X12', 'First_Y12', 'First_P12','First_X13', 'First_Y13', 'First_P13','First_X14', 'First_Y14', 'First_P14','First_X15', 'First_Y15', 'First_P15','First_X16', 'First_Y16', 'First_P16','First_X17', 'First_Y17', 'First_P17','First_X18', 'First_Y18', 'First_P18','First_X19', 'First_Y19', 'First_P19','First_X20', 'First_Y20', 'First_P10','First_X21', 'First_Y21', 'First_P21','First_X22', 'First_Y22', 'First_P22','First_X23', 'First_Y23', 'First_P23','First_X24', 'First_Y24', 'First_P24', 'Second_X0', 'Second_Y0', 'Second_P0','Second_X1', 'Second_Y1', 'Second_P1','Second_X2', 'Second_Y2', 'Second_P2','Second_X3', 'Second_Y3', 'Second_P3','Second_X4', 'Second_Y4', 'Second_P4','Second_X5', 'Second_Y5', 'Second_P5','Second_X6', 'Second_Y6', 'Second_P6','Second_X7', 'Second_Y7', 'Second_P7','Second_X8', 'Second_Y8', 'Second_P8','Second_X9', 'Second_Y9', 'Second_P9','Second_X10', 'Second_Y10', 'Second_P10','Second_X11', 'Second_Y11', 'Second_P11','Second_X12', 'Second_Y12', 'Second_P12','Second_X13', 'Second_Y13', 'Second_P13','Second_X14', 'Second_Y14', 'Second_P14','Second_X15', 'Second_Y15', 'Second_P15','Second_X16', 'Second_Y16', 'Second_P16','Second_X17', 'Second_Y17', 'Second_P17','Second_X18', 'Second_Y18', 'Second_P18','Second_X19', 'Second_Y19', 'Second_P19','Second_X20', 'Second_Y20', 'Second_P10','Second_X21', 'Second_Y21', 'Second_P21','Second_X22', 'Second_Y22', 'Second_P22','Second_X23', 'Second_Y23', 'Second_P23','Second_X24', 'Second_Y24', 'Second_P24', 'Third_X0', 'Third_Y0', 'Third_P0','Third_X1', 'Third_Y1', 'Third_P1','Third_X2', 'Third_Y2', 'Third_P2','Third_X3', 'Third_Y3', 'Third_P3','Third_X4', 'Third_Y4', 'Third_P4','Third_X5', 'Third_Y5', 'Third_P5','Third_X6', 'Third_Y6', 'Third_P6','Third_X7', 'Third_Y7', 'Third_P7','Third_X8', 'Third_Y8', 'Third_P8','Third_X9', 'Third_Y9', 'Third_P9','Third_X10', 'Third_Y10', 'Third_P10','Third_X11', 'Third_Y11', 'Third_P11','Third_X12', 'Third_Y12', 'Third_P12','Third_X13', 'Third_Y13', 'Third_P13','Third_X14', 'Third_Y14', 'Third_P14','Third_X15', 'Third_Y15', 'Third_P15','Third_X16', 'Third_Y16', 'Third_P16','Third_X17', 'Third_Y17', 'Third_P17','Third_X18', 'Third_Y18', 'Third_P18','Third_X19', 'Third_Y19', 'Third_P19','Third_X20', 'Third_Y20', 'Third_P10','Third_X21', 'Third_Y21', 'Third_P21','Third_X22', 'Third_Y22', 'Third_P22','Third_X23', 'Third_Y23', 'Third_P23','Third_X24', 'Third_Y24', 'Third_P24'])
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