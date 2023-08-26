from Funcs.LDA import LDA
from Funcs.KNN import KNN
from Funcs.QDA import QDA
from Funcs.SVM import SVM_LIN, SVM_RBF, SVM_POLY
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

###  remove the # if you want to see plots
plot = False  + True

if __name__ == '__main__':
    # Load Dataset
    data = pd.read_csv('Dataset/wine.csv')
    L = data.iloc[:, -1]
    L = list(L)
    L = np.asarray(list(L))
    X = data.iloc[:, 1:-1]
    columns = list(X.columns)
    # X = X.values

    # Split Into Training & Test
    X_train, X_test, L_train, L_test = train_test_split(X, L, test_size=0.3)

    # Standardization
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    results = {'KNN': KNN(X_train, L_train, X_test, L_test),
               'QDA': QDA(X_train, L_train, X_test, L_test),
               'LDA': LDA(X_train, L_train, X_test, L_test),
               'SVM_LIN': SVM_LIN(X_train, L_train, X_test, L_test),
               'SVM_RBF': SVM_RBF(X_train, L_train, X_test, L_test),
               'SVM_POLY': SVM_POLY(X_train, L_train, X_test, L_test)}
    
    results = {k: v*100 for k, v in results.items()}
    
    print('LDA:     ', results['LDA'])
    print('QDA:     ', results['QDA'])
    print('KNN:     ', results['KNN'])
    print('SVM_LIN: ', results['SVM_LIN'])
    print('SVM_RBF: ', results['SVM_RBF'])
    print('SVM_POLY:', results['SVM_POLY'])

    # Plotting
    if plot:
        print('Plotting...')
        Reds = data[data['type'] == 'red']
        Whites = data[data['type'] == 'white']

        from itertools import permutations
        columns.remove('quality')
        perm = list(permutations(columns, 2))

        # removing duplicates => [a,b] == [b,a]
        for i in perm:
            if i[::-1] in perm:
                perm.remove(i)

        print(len(perm))

        fig = plt.figure()
        index = 0
        for i in perm:
            ax = fig.add_subplot(5, 9, index + 1)
            ax.scatter(Reds[i[0]], Reds[i[1]], c='red', s=5, label=i)
            ax.scatter(Whites[i[0]], Whites[i[1]], c='black', s=1, label=i)
            index += 1
            # you can uncomment these lines to
            # view name of each axis
            # plt.title(i[0] + '/' + i[1])
            ax.set_xlabel(i[0])
            ax.set_ylabel(i[1])
        plt.show()
