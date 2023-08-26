def SVM_LIN(X_train, L_train, X_test, L_test):
    from sklearn.svm import SVC
    # Testing KNN
    Model = SVC(kernel='linear')
    # Train
    Model.fit(X_train, L_train)
    # Predict
    Pre = Model.predict(X_test)
    Score = Model.score(X_test, L_test)
    return Score

def SVM_RBF(X_train, L_train, X_test, L_test):
    from sklearn.svm import SVC
    # Testing KNN
    Model = SVC(kernel='rbf')
    # Train
    Model.fit(X_train, L_train)
    # Predict
    Pre = Model.predict(X_test)
    Score = Model.score(X_test, L_test)
    return Score

def SVM_POLY(X_train, L_train, X_test, L_test):
    from sklearn.svm import SVC
    # Testing KNN
    Model = SVC(kernel='poly',degree=2)
    # Train
    Model.fit(X_train, L_train)
    # Predict
    Pre = Model.predict(X_test)
    Score = Model.score(X_test, L_test)
    return Score

