import pickle
from azureml.core import Workspace
from azureml.core.run import Run
import os
from azureml.core import Workspace, Experiment, Datastore
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from azureml.core.authentication import AzureCliAuthentication
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


import numpy as np
import json
import subprocess
from typing import Tuple, List
import argparse
import pandas as pd



# run_history_name = 'devops-ai'
# os.makedirs('./outputs', exist_ok=True)
# #ws.get_details()
# Start recording results to AML
# run = Run.start_logging(workspace = ws, history_name = run_history_name)
# run = Run.get_submitted_run()

def getRuntimeArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--step_output', type=str)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    return args

def main():
    args = getRuntimeArgs()
    run = Run.get_context()

    credit_data_df = pd.read_csv(os.path.join(args.data_path, 'german_credit_data.csv'))
    clf = model_train(credit_data_df, run)

    #copying to "outputs" directory, automatically uploads it to Azure ML
    # output_dir = './outputs/'
    # os.makedirs(output_dir, exist_ok=True)
    # joblib.dump(value=clf, filename=os.path.join(output_dir, 'model.pkl'))
    # model_name = args.model_name
    # data_store = args.step_output
    # Pass model file to next step
    ws = run.experiment.workspace
    cli_auth = AzureCliAuthentication()
    datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                        datastore_name='modelstore',
                                                        container_name='model',
                                                        account_name='amldemo2398627300',
                                                        account_key='Q88W/8T4fAq4dQnU7IjTKsTArZZylmeSbW+PFFr+XL87cbdwLzkmqN+LHzq/W1sN6/hkd2EqFj5+JM9Iln4lUg==',
                                                        create_if_not_exists=True)

    # ws.set_default_datastore('ypfdatastore')
    # # datastore = ws.get_default_datastore()
    # # print(datastore)


    os.makedirs("./model", exist_ok=True)
    # model_output_path = os.path.join(step_output_path, model_name)
    joblib.dump(value=clf, filename="./model/model.pkl")
    # data_store = Datastore.get_default(ws)
    print(f"tags now present for run: {run.tags}")
    print("****************************************")
    print(os.listdir("./"))
    print(os.listdir("./model"))
    print(datastore.name)
    print("****************************************")

    datastore.upload(src_dir="./model", target_path="model/", overwrite=True)
    # Also upload model file to run outputs for history
    # os.makedirs('outputs', exist_ok=True)
    # output_path = os.path.join('outputs', model_output_path)
    # joblib.dump(value=clf, filename=output_path)

    run.tag("run_type", value="train")


def model_train(ds_df, run):
    ds_df.drop("Sno", axis=1, inplace=True)

    y_raw = ds_df['Risk']
    X_raw = ds_df.drop('Risk', axis=1)

    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['int64', 'float']).columns

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
        ('onehotencoder', OneHotEncoder(categories='auto', sparse=False))])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    feature_engineering_pipeline = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")

    # Encode Labels
    le = LabelEncoder()
    encoded_y = le.fit_transform(y_raw)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, encoded_y, test_size=0.20, stratify=encoded_y,
                                                        random_state=42)

    # Create sklearn pipeline
    lr_clf = Pipeline(steps=[('preprocessor', feature_engineering_pipeline),
                             ('classifier', LogisticRegression(solver="lbfgs"))])
    # Train the model
    lr_clf.fit(X_train, y_train)

    # Capture metrics
    train_acc = lr_clf.score(X_train, y_train)
    test_acc = lr_clf.score(X_test, y_test)
    print("Training accuracy: %.3f" % train_acc)
    print("Test data accuracy: %.3f" % test_acc)

    # Log to Azure ML
    run.log('Train accuracy', train_acc)
    run.log('Test accuracy', test_acc)

    return lr_clf


if __name__ == "__main__":
    main()