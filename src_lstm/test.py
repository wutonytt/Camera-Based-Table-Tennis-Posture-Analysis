import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def test(model, rightTestData, leftTestData, modelRight, modelLeft):
    Xright = rightTestData.iloc[:, 1:-1]
    yright_actual = rightTestData.iloc[:, -1].values
    
    if model == 'LSTM':
        # Convert to dim-3
        Xright_np = Xright.to_numpy()
        Xright_np = np.reshape(Xright_np, (Xright_np.shape[0], Xright_np.shape[1], 1))
        yright_pred = modelRight.predict(Xright_np)
    
    elif model == 'SVM':
        yright_pred = modelRight.predict(Xright)
        
    print('Right', classification_report(yright_actual, yright_pred))
    
    
    Xleft = leftTestData.iloc[:, 1:-1]
    yleft_actual = leftTestData.iloc[:, -1].values
    
    if model == 'LSTM':
        # Convert to dim-3
        Xleft_np = Xleft.to_numpy()
        Xleft_np = np.reshape(Xleft_np, (Xleft_np.shape[0], Xleft_np.shape[1], 1))
        yleft_pred = modelLeft.predict(Xleft_np)
        
    elif model == 'SVM':
        yleft_pred = modelLeft.predict(Xleft)
        
    print('Left', classification_report(yleft_actual, yleft_pred))

    yright_pred = np.where(yright_pred == 1, 'Fore', 'Back')
    yleft_pred = np.where(yleft_pred == 1, 'Fore', 'Back')
    
    y = pd.concat([pd.Series(yright_pred), pd.Series(yleft_pred)], ignore_index=True)
    testData = pd.concat([rightTestData, leftTestData], ignore_index=True)

    return testData