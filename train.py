# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    is_test = sys.argv[3] == 'y' if len(sys.argv) > 3 else False
    
    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
        
        if is_test:
            import json

            # Create some files to preserve as artifacts
            features = "rooms, zipcode, median_price, school_rating, transport"
            data = {"state": "TX", "Available": 25, "Type": "Detached"}

            # Create couple of artifact files under the directory "data"
            os.makedirs("data", exist_ok=True)
            with open("data/data.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            with open("data/features.txt", 'w') as f:
                f.write(features)
            
            mlflow.log_artifacts("data", artifact_path="states")
            
            mlflow.log_artifact(wine_path)
            
            mlflow.log_params({"alpha":alpha, "l1_ratio":l1_ratio})
            mlflow.log_metrics({"rmse":rmse, "r2":r2, "mae":mae})
            
            mlflow.log_text("text test", "testtext.txt")
            
    if is_test:
        model_uri = "runs:/{}/model".format(run.info.run_id)
        mv = mlflow.register_model(model_uri, "ElasticNetRegressionModel")
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
