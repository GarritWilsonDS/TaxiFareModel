from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipeline = Pipeline([
            ("dist_trans", DistanceTransformer()),
            ("scaler", StandardScaler())
        ])

        time_pipe = Pipeline([
            ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ])

        preproc_pipe = ColumnTransformer([
            ("dist", dist_pipeline, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude']),
            ("time", time_pipe, ["pickup_datetime"])
            ])

        final_pipe = Pipeline([
            ("preproc", preproc_pipe),
            ("linear_model", LinearRegression())
        ])

        return final_pipe

    def run(self):
        """set and train the pipeline"""
        pipe = self.set_pipeline()

        pipe.fit(self.X,self.y)

        return pipe


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)

        rmse = np.sqrt(((y_pred - y_test)**2).mean())

        return rmse


if __name__ == "__main__":
    data = get_data()
    cleaned_data = clean_data(data)

    X = data.drop("fare_amount", axis= 1)
    y = data["fare_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)

    trainer = Trainer(X_train, y_train)

    trainer.run()

    rmse= trainer.evaluate(X_test, y_test)

    print(f"rmse = {rmse}")
