import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

categorical_columns = ['Sex', 'Embarked', 'Cabin']
def encode_categorical_columns(df, categorical_columns):
    df = df.copy()
    for column in categorical_columns:
        le = LabelEncoder()
        column_name = column + '_encoded'
        df[column_name] = le.fit_transform(df[column])
    return df

df_encoded = encode_categorical_columns(df, categorical_columns)
df_encoded.head()

feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_encoded', 'Embarked_encoded', 'Cabin_encoded']
target_column = 'Survived'
def decision_tree_pipeline(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']).rename(columns={0: 'Not Survived', 1: 'Survived'}, index={0: 'Not Survived', 1: 'Survived'})
    print('Confusion Matrix:\n', confusion_matrix)
    print(f'Decision Tree Classifier Accuracy: {accuracy:.4f}')
    return model, scaler

decision_tree_model, scaler = decision_tree_pipeline(df_encoded, feature_columns, target_column)
