def LDA(X_train, L_train, X_test, L_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # Testing KNN
    Model = LinearDiscriminantAnalysis()

    # Train
    Model.fit(X_train, L_train)

    # Predict
    Pre = Model.predict(X_test)
    Score = Model.score(X_test, L_test)

    # print('LDA Model.Score:', Score)
    return Score

