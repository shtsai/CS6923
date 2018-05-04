import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

def extract_month(row):
    '''Extract month from FL_DATE'''
    date = row["FL_DATE"].split("-")
    return int(date[1])

def extract_day(row):
    '''Extract day from FL_DATE'''
    date = row["FL_DATE"].split("-")
    return int(date[2])

def extract_hour(row):
    '''Extract hour from CRS_DEP_TIME'''
    hour = row["CRS_DEP_TIME"] // 100
    return hour

def clean_column(column):
    def _clean_column(row):
        return int(row[column].replace(',', ''))
    return _clean_column

def compute_speed(row):
    '''Compute average speed of the flight'''
    speed = row["DISTANCE"] / row["ACTUAL_ELAPSED_TIME"]
    return speed

def one_hot_encode(df, column):
    '''Perform one-hot encoding on the given column, return new dataframe and a set containing column values'''
    one_hot = pd.get_dummies(df[column], prefix=column)
    values = set(one_hot)
    df = df.join(one_hot)
    return df, values

def standardize_feature(training, testing, column):
    '''Standardize features by removing the mean and scaling to unit variance'''
    scaler = StandardScaler()
    scaler.fit(training[column].values.reshape(-1, 1))
    training[column] = scaler.transform(training[column].values.reshape(-1, 1))
    testing[column] = scaler.transform(testing[column].values.reshape(-1, 1))
    return training, testing

def preprocess(training, testing):
    '''Proprocess training and testing data, so that they have the same schema'''
    # extract labels
    labels = training["ARR_DELAY"]

    # extract month and day
    training["MONTH"] = training.apply(extract_month, axis=1)
    training["DAY"] = training.apply(extract_day, axis=1)
    training["HOUR"] = training.apply(extract_hour, axis=1)

    # one hot encoding for DAY_OF_WEEK, AIRLINE_ID, ORIGIN and DEST
    training, DAY_OF_WEEK = one_hot_encode(training, "DAY_OF_WEEK")
    training, MONTH = one_hot_encode(training, "MONTH")
    training, DAY = one_hot_encode(training, "DAY")
    training, HOUR = one_hot_encode(training, "HOUR")
    training, AIRLINE_ID = one_hot_encode(training, "AIRLINE_ID")
    training, ORIGIN = one_hot_encode(training, "ORIGIN")
    training, DEST = one_hot_encode(training, "DEST")


    # clean DISTANCE add SPEED attribute
    training["DISTANCE"] = training.apply(clean_column("DISTANCE"), axis=1)
    training["SPEED"] = training.apply(compute_speed, axis=1)

    # drop unneeded attributes
    training = training.drop(columns=["DAY_OF_WEEK", "FL_DATE", "MONTH", "DAY", "HOUR","AIRLINE_ID",
                                      "UNIQUE_CARRIER", "FL_NUM", "ORIGIN_CITY_MARKET_ID",
                                      "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID",
                                      "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "CRS_DEP_TIME", "DISTANCE_GROUP",
                                      "FIRST_DEP_TIME", "ARR_DELAY", "UID"])

    # PREPROCESSING TESTING DATA
    # extract month and day
    testing["MONTH"] = testing.apply(extract_month, axis=1)
    testing["DAY"] = testing.apply(extract_day, axis=1)
    testing["HOUR"] = testing.apply(extract_hour, axis=1)

    # one hot encoding for DAY_OF_WEEK, AIRLINE_ID, ORIGIN and DEST
    testing, DAY_OF_WEEK = one_hot_encode(testing, "DAY_OF_WEEK")
    testing, MONTH = one_hot_encode(testing, "MONTH")
    testing, DAY = one_hot_encode(testing, "DAY")
    testing, HOUR = one_hot_encode(testing, "HOUR")
    testing, AIRLINE_ID = one_hot_encode(testing, "AIRLINE_ID")
    testing, ORIGIN = one_hot_encode(testing, "ORIGIN")
    testing, DEST = one_hot_encode(testing, "DEST")

    # clean DISTANCE add SPEED attribute
    testing["DISTANCE"] = testing.apply(clean_column("DISTANCE"), axis=1)
    testing["SPEED"] = testing.apply(compute_speed, axis=1)

    # drop unneeded attributes
    testing = testing.drop(columns=["DAY_OF_WEEK", "FL_DATE", "MONTH", "DAY", "HOUR","AIRLINE_ID",
                                      "UNIQUE_CARRIER", "FL_NUM", "ORIGIN_CITY_MARKET_ID",
                                      "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID",
                                      "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "CRS_DEP_TIME", "DISTANCE_GROUP",
                                      "FIRST_DEP_TIME", "UID"])

    cols = training.columns
    training, testing = training.align(testing, join='outer', axis=1, fill_value=0)
    training = training[cols]
    testing = testing[cols]

    # standardize features by removing the mean and scaling to unit variance
    training, testing = standardize_feature(training, testing, "TAXI_OUT")
    training, testing = standardize_feature(training, testing, "TAXI_IN")
    training, testing = standardize_feature(training, testing, "ACTUAL_ELAPSED_TIME")
    training, testing = standardize_feature(training, testing, "DISTANCE")
    training, testing = standardize_feature(training, testing, "SPEED")

    return training, labels, testing

def error(x, y):
    N = len(x)
    return np.sum(np.square(x - y)) / N

def ridge_predict(training, labels, testing):
    # First perform cross-validation to find the best value for alpha
    best_alpha = ridge_cv(training, labels, [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    # Then perform ridge regression
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(training, labels)
    return ridge.predict(training), ridge.predict(testing)

def ridge_cv(training, labels, alphas):
    # Perform 10 fold cross-validation to find the best value for alpha
    kf = KFold(n_splits=10, shuffle=True)

    min_error = float("inf")
    best_alpha = None
    for a in alphas:
        ridge = Ridge(alpha=a)
        ridge_error = 0.0
        for tr_index, te_index in kf.split(training):
            kf_training = training.iloc[tr_index]
            kf_labels = labels.iloc[tr_index]
            kf_testing = training.iloc[te_index]
            kf_testing_labels = labels.iloc[te_index]

            ridge.fit(kf_training, kf_labels)
            ridge_prediction = ridge.predict(kf_testing)
            ridge_error += error(ridge_prediction, kf_testing_labels)
        ridge_error /= kf.n_splits
        print(a, ridge_error)
        if ridge_error < min_error:
            min_error = ridge_error
            best_alpha = a
    return best_alpha


def nn_10fold_CV(training, labels):
    '''Perform 10-fold cross validation on different configurations of neural networks.
       Return the configuration with lowest error'''
    kf = KFold(n_splits=10, shuffle=True)

    nns = []
    ## TODO add here
    nns.append(MLPRegressor(hidden_layer_sizes=(4096, 1024, 256, 1), activation='logistic', solver='sgd',
                            max_iter=100000, early_stopping=True, learning_rate='adaptive',learning_rate_init=0.5))
    nns.append(MLPRegressor(hidden_layer_sizes=(4096, 1024, 256, 1), activation='relu', solver='adam',
                            max_iter=100000, early_stopping=True, learning_rate_init=0.5))
    nns.append(MLPRegressor(hidden_layer_sizes=(4096, 1024, 256, 1), activation='relu', solver='sgd',
                            max_iter=100000, early_stopping=True, learning_rate='adaptive',learning_rate_init=0.5))

    min_error = float("inf")
    best_nn = None

    for nn in nns:
        nn_error = 0.0
        for tr_index, te_index in kf.split(training):
            kf_training = training.iloc[tr_index]
            kf_labels = labels.iloc[tr_index]
            kf_testing = training.iloc[te_index]
            kf_testing_labels = labels.iloc[te_index]

            nn.fit(kf_training, kf_labels)
            nn_prediction = nn.predict(kf_testing)
            nn_error += error(nn_prediction, kf_testing_labels)
        nn_error /= kf.n_splits
        print(nn)
        print(nn_error)
        if nn_error < min_error:
            min_error = nn_error
            best_nn = nn
    return best_nn

def neural_net_predict(training, labels, testing):
    '''
    This function performs neural net regression.
    The function will first run cross-validation on different configurations of the neural net,
    and determine the best configuration to use.
    Here the function call to nn_10fold_CV() is commented out because this step is very time comsuming.
    Instead, I hardcode the best configuration I found from the cross validation.
    '''
    best_nn = nn_10fold_CV(training, labels)
    nn = MLPRegressor(hidden_layer_sizes=best_nn.hidden_layer_sizes, activation=best_nn.activation,
                      solver=best_nn.solver, max_iter=best_nn.max_iter, early_stopping=best_nn.early_stopping,
                      learning_rate=best_nn.learning_rate, learning_rate_init=best_nn.learning_rate_init)
    #nn = MLPRegressor(hidden_layer_sizes=(10000, 1000), activation='relu', solver='adam', max_iter=100000,
    #                  learning_rate_init=0.0005, alpha=0.00000, early_stopping=True)
    nn.fit(training, labels)
    return nn.predict(training), nn.predict(testing)

def main():
    # Read training and testing data
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    training = pd.read_csv(training_file)
    testing = pd.read_csv(testing_file)

    # preprocess training data
    training_id = training["UID"]
    testing_id = testing["UID"]
    training, labels, testing = preprocess(training, testing)

    # Ridge Regression
    #ridge_training_prediction, ridge_testing_prediction = ridge_predict(training, labels, testing)
    #ridge_error = error(ridge_training_prediction, labels)
    #print(ridge_error)

    # Neural Net Regression
    nn_training_prediction, nn_testing_prediction = neural_net_predict(training, labels, testing)
    print(nn_training_prediction)
    nn_error = error(nn_training_prediction, labels)
    print(nn_error)
    print(nn_error)



if __name__ == "__main__":
    main()
