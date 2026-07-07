def kfold_multi_models(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results_logistic = []
    results_svc = []
    results_random_forest = []
    results_decision_tree = []
    results = [results_logistic, results_svc, results_random_forest, results_decision_tree]
    model_names = ['Logistic Regression', 'SVC', 'Random Forest', 'Decision Tree']
    max_score = 0.0
    max_model = None
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logistic_model = LogisticRegression(max_iter=200)
        svc_model = SVC(C=1.0, kernel='linear', probability=True)
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        models = [logistic_model, svc_model, random_forest_model, decision_tree_model]
        for i, (model, _) in enumerate(zip(models, results)):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if max_score < accuracy:
                max_score = accuracy
                max_model = model_names[i]
            results[i].append(accuracy)
    print(f"Best Model: {max_model} with Accuracy: {max_score:.4f}")
    generally_best_model_index = np.argmax([np.mean(result) for result in results])
    print(f"Generally Best Model: {model_names[generally_best_model_index]} with Average Accuracy: {np.mean(results[generally_best_model_index]):.4f}")
    return results, model_names

kfold_multi_models(iris.data, iris.target)
