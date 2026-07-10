from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
def compare_regressions(X, y):
    """"Adopts regularization methods to compare"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    ## Initial scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # L1 norm penalty
    lasso = Lasso(alpha=0.1)
    # L2 norm -enalty
    ridge = Ridge(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    y_pred_ridge = ridge.predict(X_test_scaled)
    print(f"Lasso Coefficient: {r2_score(y_pred_lasso, y_test)}")
    print(f"Ridge Coefficient: {r2_score(y_pred_ridge, y_test)}")
    print(f"Lasso socre: {lasso.score(X_test_scaled, y_test)} vs Ridge score: {ridge.score(X_test_scaled, y_test)}")
compare_regressions(X, y)