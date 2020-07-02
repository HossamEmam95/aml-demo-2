import argparse
import json
import pandas as pd
import os

from azureml.core.run import Run
from azureml.core import Datastore
from azureml.core.authentication import AzureCliAuthentication

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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

    ws = run.experiment.workspace
    AzureCliAuthentication()

    # load configs
    with open("aml_config/config.json", "r") as f:
        configs = json.load(f)

    # Register Data store to save the model
    datastore_data = configs['DataStore_Data']
    datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                        datastore_name=datastore_data["datastore_name"],
                                                        container_name=datastore_data["container_name"],
                                                        account_name=datastore_data["account_name"],
                                                        account_key=datastore_data["account_key"],
                                                        create_if_not_exists=True)
    print(datastore.name)

    # write model and upload it to data store
    model_data = configs["Model_Data"]
    os.makedirs("./model", exist_ok=True)
    joblib.dump(value=clf, filename="./model/" + model_data["model_name"])
    print(f"tags now present for run: {run.tags}")
    datastore.upload(src_dir="./model", target_path=datastore_data["target_path"], overwrite=True)
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