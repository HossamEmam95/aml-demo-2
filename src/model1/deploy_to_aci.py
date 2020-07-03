import json

from azureml.core.run import Run
from azureml.core.model import Model
from azureml.core import Webservice
from azureml.exceptions import WebserviceException
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

run = Run.get_context()
ws = run.experiment.workspace

with open("aml_config/config.json", "r") as f:
    configs = json.load(f)
model_data = configs["Model_Data"]

model = Model(ws, model_data["model_name"])
print(model.name,  model.version)

AzureCliAuthentication()

service_name = 'aml-demo-aci-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

conda_deps = CondaDependencies(conda_dependencies_file_path="src/model1/aml_config/inference-conda.yml")
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps

inf_config = InferenceConfig(entry_script='src/model1/score.py', environment=myenv)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inf_config,
                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)