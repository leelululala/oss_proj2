import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler






def sort_dataset(dataset_df):
    kbo = dataset_df.sort_values(by=['year'])
    return kbo


def mul_0_001(x):
    return x * 0.001


def split_dataset(dataset_df):
    dataset_df['salary'] = dataset_df['salary'].apply(mul_0_001)
    y = dataset_df['salary']
    x = dataset_df.drop(columns="salary", axis=1)
    x_train = x.iloc[:1718]
    x_test = x.iloc[1718:]
    y_train = y.iloc[:1718]
    y_test = y.iloc[1718:]
    return x_train, x_test, y_train, y_test



def extract_numerical_cols(dataset_df):
    num_col = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    return num_col


def train_predict_decision_tree(X_train, Y_train, X_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, Y_train)
    predict = dtr.predict(X_test)
    return predict



def train_predict_random_forest(X_train, Y_train, X_test):
    rf_rg = RandomForestRegressor()
    rf_rg.fit(X_train, Y_train)
    predict = rf_rg.predict(X_test)
    return predict



def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pipe.fit(X_train,Y_train)
    predict = svm_pipe.predict(X_test)
    return predict




def calculate_RMSE(labels, predictions):
     return np.sqrt(np.mean((predictions-labels)**2))






if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))