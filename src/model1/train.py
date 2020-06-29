import pickle
from azureml.core import Workspace
from azureml.core.run import Run
import os
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
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
    parser.add_argument('--step_output', type=str),
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    return args

def main():
    args = getRuntimeArgs()
    run = Run.get_context()

    step_output_path = args.step_output
    model_name = args.model_name

    credit_data_df = pd.read_csv(os.path.join(args.data_path, 'german_credit_data.csv'))
    clf = model_train(credit_data_df, run)

    # #copying to "outputs" directory, automatically uploads it to Azure ML
    # output_dir = './outputs/'
    # os.makedirs(output_dir, exist_ok=True)
    # joblib.dump(value=clf, filename=os.path.join(output_dir, 'model.pkl'))

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_name)
    joblib.dump(value=clf, filename=model_output_path)

    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    joblib.dump(value=clf, filename=output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")


def model_train(ds_df, run):
    X, y = load_diabetes(return_X_y=True)
    columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

    print("Running train.py")

    # Randomly pic alpha
    alphas = np.arange(0.0, 1.0, 0.05)
    alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
    print(alpha)
    run.log("alpha", alpha)
    reg = Ridge(alpha=alpha)
    reg.fit(data["train"]["X"], data["train"]["y"])
    preds = reg.predict(data["test"]["X"])
    run.log("mse", mean_squared_error(preds, data["test"]["y"]))


    # # Save model as part of the run history
    # model_name = "sklearn_regression_model.pkl"
    # # model_name = "."

    # with open(model_name, "wb") as file:
    #     joblib.dump(value=reg, filename=model_name)

    # # upload the model file explicitly into artifacts
    # run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
    # print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
    # dirpath = os.getcwd()
    # print(dirpath)

    # # model explain start
    # from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
    # from azureml.core.run import Run
    # from interpret.ext.blackbox import TabularExplainer

    # run = Run.get_context()
    # client = ExplanationClient.from_run(run)

    # # explain predictions on your local machine
    # # "features" and "classes" fields are optional
    # explainer = TabularExplainer(reg, 
    #                             X_train)

    # # explain overall model predictions (global explanation)
    # global_explanation = explainer.explain_global(X_test)

    # # uploading global model explanation data for storage or visualization in webUX
    # # the explanation can then be downloaded on any compute
    # # multiple explanations can be uploaded
    # client.upload_model_explanation(global_explanation, comment='global explanation: all features')
    # # or you can only upload the explanation object with the top k feature info
    # #client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')
    # # model explain end


    # # register the model
    # # run.log_model(file_name = model_name)
    # # print('Registered the model {} to run history {}'.format(model_name, run.history.name))


    # print("Following files are uploaded ")
    # print(run.get_file_names())
    # run.complete()
    return reg


if __name__ == "__main__":
    main()