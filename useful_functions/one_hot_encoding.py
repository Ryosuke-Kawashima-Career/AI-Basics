from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
def one_hot_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Returns a new df with some one hot encoding columns"""
    ohe = OneHotEncoder(sparse_output=False)
    encoded_data = ohe.fit_transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out(columns),
        index=df.index
    )
    df_encoded = pd.concat([df, encoded_df], axis=1)
    df_encoded.drop(columns=columns, inplace=True)
    return df_encoded
