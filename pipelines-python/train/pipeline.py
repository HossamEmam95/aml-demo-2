import json
import argparse
import azureml.core
from azureml.core import Workspace, Datastore, Dataset, RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

print("Azure ML SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser("deploy_training_pipeline")
parser.add_argument("--pipeline_name", type=str, help="Name of the pipeline that will be deployed", dest="pipeline_name", required=True)
parser.add_argument("--build_number", type=str, help="Build number", dest="build_number", required=False)
parser.add_argument("--dataset", type=str, help="Default dataset, referenced by name", dest="dataset", required=True)
parser.add_argument("--runconfig", type=str, help="Path to runconfig for pipeline", dest="runconfig", required=True)
parser.add_argument("--source_directory", type=str, help="Path to model training code", dest="source_directory", required=True)
args = parser.parse_args()
print(f'Arguments: {args}')

print('Connecting to workspace')
ws = Workspace.from_config()
print(f'WS name: {ws.name}\nRegion: {ws.location}\nSubscription id: {ws.subscription_id}\nResource group: {ws.resource_group}')

print('Loading runconfig for pipeline')
runconfig = RunConfiguration.load(args.runconfig)

print('Loading dataset')    
training_dataset = Dataset.get_by_name(ws, args.dataset)

# Parametrize dataset input to the pipeline
training_dataset_parameter = PipelineParameter(name="training_dataset", default_value=training_dataset)
training_dataset_consumption = DatasetConsumptionConfig("training_dataset", training_dataset_parameter).as_mount()

pipeline_data = PipelineData(
    "pipeline_data", datastore=ws.get_default_datastore()
)
data_store = Datastore.get_default(ws)

with open("src/model1/aml_config/config.json", "r") as f:
    configs = json.load(f)

datastore_data = configs['DataStore_Data']
model_data = configs["Model_Data"]

model_name_param = PipelineParameter(name="model_name", default_value=model_data["model_name"])

train_step = PythonScriptStep(name="train-step",
                        runconfig=runconfig,
                        source_directory=args.source_directory,
                        script_name=runconfig.script,
                        arguments=['--data-path', training_dataset_consumption, "--step_output", "data_store", "--model_name", model_data["model_name"], ],
                        inputs=[training_dataset_consumption],
                        outputs=[pipeline_data],
                        allow_reuse=False)

register_step = PythonScriptStep(name="Register Model",
                        runconfig=runconfig,
                        script_name="./register.py",
                        source_directory=args.source_directory,
                        inputs=[pipeline_data],
                        arguments=["--model_name", model_name_param, "--step_input", "data_store", "--model_name", model_data["model_name"], ],  # NOQA: E501
                        allow_reuse=False)


register_step.run_after(train_step)

steps = [train_step, register_step]

print('Creating and validating pipeline')
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

print('Publishing pipeline')
published_pipeline = pipeline.publish(args.pipeline_name)

# Output pipeline_id in specified format which will convert it to a variable in Azure DevOps
print(f'##vso[task.setvariable variable=pipeline_id]{published_pipeline.id}')
