from datetime import datetime
import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    df = df.copy()

    return df.drop(columns_to_drop, axis=1)


def outlier_removal(df):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    df = df.copy()

    boundaries = calculate_outliers(df['year'])
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df


def feature_engineering(df):
    def short_model(x):
        import pandas
        if not pandas.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df.drop(['model', 'year'], axis=1)


def main():
    print('Price Category Prediction Pipeline')
    df = pd.read_csv('data/homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df.price_category

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_pipe = None
    best_accuracy = 0.0
    for model in models:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        column_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ])

        preprocessor = Pipeline([
            ('filter', FunctionTransformer(filter_data)),
            ('outlier_remover', FunctionTransformer(outlier_removal)),
            ('feature_engineering', FunctionTransformer(feature_engineering)),
            ('column_transformer', column_transformer)
        ])

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_name: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_accuracy:
            best_accuracy = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_accuracy:.4f}')

    best_pipe.fit(X, y)

    with open('cars_pipe.pkl', 'wb') as f:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car price class prediction model',
                'author': 'Yaroslav Zamuruyev',
                'version': 1,
                'date': datetime.utcnow(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_accuracy
            }
        }, f)


if __name__ == "__main__":
    main()
