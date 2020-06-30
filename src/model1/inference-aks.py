from azureml.core.environment import Environment
from azureml.core.run import Run
from azureml.core.model import Model
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice, AksWebservice

run = Run.get_context()
ws = run.experiment.workspace

from azureml.core.conda_dependencies import CondaDependencies

conda_deps = CondaDependencies(conda_dependencies_file_path="aml_config/inference-conda.yml")
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps

# myenv = Environment.from_conda_specification(name='env', file_path='aml_config/inference-config.yml')

# myenv.register(workspace=ws)

model = Model(ws, "model.pkl")

# Prepare AKS
aks_service_name = "aml-demo-python"
compute_list = ws.compute_targets()
aks_target, = (c for c in compute_list if c.name == aks_service_name)

# create if not exist
if not aks_target:
    prov_config = AksCompute.provisioning_configuration()

    aks_target = ComputeTarget.create(
            workspace=ws, name=aks_service_name, provisioning_configuration=prov_config)

    aks_target.wait_for_completion(show_output=True)


inference_config = InferenceConfig(entry_script='./score.py', environment=myenv)

aks_config = AksWebservice.deploy_configuration()

aks_service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
