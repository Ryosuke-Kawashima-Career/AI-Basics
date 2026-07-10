from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
def grid_cross_validation(X: pd.DataFrame, y: pd.Series) -> dict:
    model_configs = {
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(max_iter=100),
            'params': {
                'C': [0.1, 1, 10]
            }
        },
        'naive_bayesian': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.0, 0.5, 1.0]
            }
        }
    }
    best_models = {}
    for name, config in model_configs.items():
        grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=5)
        grid_search.fit(X, y)
        best_models[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    return best_models

result = grid_cross_validation(df, y)
print(result)
