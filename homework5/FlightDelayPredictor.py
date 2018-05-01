import sys
import pandas as pd

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
    '''Remove special characters in the given column'''
    def _clean_column(row):
        '''Actual function to clean column'''
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

def main():
    # Read training and testing data
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    training = pd.read_csv(training_file)
    testing = pd.read_csv(testing_file)

    # PREPROCESSING TRAINING DATA
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
                                      "CARRIER", "FL_NUM", "ORIGIN_CITY_MARKET_ID",
                                      "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID",
                                      "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "CRS_DEP_TIME", "DISTANCE_GROUP",
                                      "FIRST_DEP_TIME"])

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
                                      "CARRIER", "FL_NUM", "ORIGIN_CITY_MARKET_ID",
                                      "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID",
                                      "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "CRS_DEP_TIME", "DISTANCE_GROUP",
                                      "FIRST_DEP_TIME"])

    cols = training.columns
    ## TODO need to seperate ARR_DELAY from training
    training, testing = training.align(testing, join='outer', axis=1, fill_value=0)
    training = training[cols]
    testing = testing[cols]


if __name__ == "__main__":
    main()
