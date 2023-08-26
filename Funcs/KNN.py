
def KNN(X_train, L_train, X_test, L_test):
    from sklearn.neighbors import KNeighborsClassifier
    # Testing KNN
    Model = KNeighborsClassifier(n_neighbors=3)

    # Train
    Model.fit(X_train, L_train)

    # Predict
    Pre = Model.predict(X_test)
    Score = Model.score(X_test, L_test)

    # print('KNN Model.Score:', Score)
    return Score
