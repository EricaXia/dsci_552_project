""" 
Test models, fine tune hps, get SHAP importances, and retest/fine tune models. Check how they differ.
 """

# lr = LogisticRegression()
# param_dist = {
#     'penalty': ['l1', 'l2'],
#     'C': np.logspace(-4, 4, 20),
#     'solver': ['liblinear']
# }


def test_model(model, param_dist, type_of_model, X_train, X_test, y_train, y_test, min_count=35):
    model_rand = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                    n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)
    
    model_rand.fit(X_train, y_train)
    best_params = model_rand.best_params_
    model_1 = type_of_model(**best_params)
    score = cross_val_score(model_1, X_train, y_train)
    print("First CV score:", score)
    y_pred = model_1.predict(X_test)
    target_names = ['0: No Revenue', '1: Revenue']
    print("First classif. report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("ROC AUC Score:", roc_auc_score(y_test, model_1.predict_proba(X_test)[:, 1]))
    # get SHAP values
    X_test_sample = shap.utils.sample(X_test, nsamples=50, random_state=0)
    explainer = shap.TreeExplainer(model_1)
    shap_values = explainer.shap_values(X_test_sample)
    top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
    f_indices = list(top_inds[:, :20].flatten())
    c = Counter(f_indices)
    to_keep = []
    for idx, count in c.items():
        if count >= min_count:
            to_keep.append(idx)
    print(f"Keeping {len(to_keep)} features")
    X_train_ss = X_train.iloc[:, to_keep]
    X_test_ss = X_test.iloc[:, to_keep]
    model_rand2 = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                    n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)
    
    model_rand2.fit(X_train_ss, y_train)
    best_params2 = model_rand2.best_params_
    model_2 = type_of_model(**best_params2)
    score2 = cross_val_score(model_2, X_train_ss, y_train)
    print("Second CV score:", score2)
    y_pred2 = model_2.predict(X_test_ss)
    print("Second classif. report:")
    print(classification_report(y_test, y_pred2, target_names=target_names))
    print("Second ROC AUC Score:", roc_auc_score(y_test, model_2.predict_proba(X_test_ss)[:, 1]))

