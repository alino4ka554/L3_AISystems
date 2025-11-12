model = GaussianNB(var_smoothing=0)
model = DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=3, 
            random_state=42
        )
model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
model = SVC(C=1, kernel='rbf', gamma='auto', random_state=42)
model = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')
