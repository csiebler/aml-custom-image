
# Custom Images with Azure Machine Learning

This repo shows how to use custom Docker image in Azure Machine Learning for model training and real-time prediction.

First, build a custom image that contains all relevant libraries used by your Machine Learning code:

```console
# Connect to Azure Container Registry
az acr login --name <REGISTRY_NAME>
docker login aml<REGISTRY_NAME>.azurecr.io

# Build and push image locally
docker build . -t <REGISTRY_NAME>.azurecr.io/azureml-images/scikit-learn:0.23.2
docker push <REGISTRY_NAME>.azurecr.io/azureml-images/scikit-learn:0.23.2
```

For details, see the [`Dockerfile`](Dockerfile). All relevant libraries are defined in [`requirements.txt`](requirements.txt) (can be easily adapted to `conda` or `virtualenv`).

Next, you can use the image in a training job in Azure Machine Learning:

```python
from azureml.core import Workspace, Environment
from azureml.core import ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core import Experiment

ws = Workspace.from_config()

custom_env = Environment("custom_env")
custom_env.docker.enabled = True
custom_env.python.user_managed_dependencies = True
custom_env.docker.base_image = "<REGISTRY_NAME>.azurecr.io/azureml-images/scikit-learn:0.23.2"

cluster_name = "cpu-cluster"
compute_target = ComputeTarget(workspace=ws, name=cluster_name)

src = ScriptRunConfig(source_directory='./',
                      script='train.py',
                      compute_target=compute_target,
                      environment=custom_env)

run = Experiment(ws,'custom-image-training').submit(src)
run.wait_for_completion(show_output=True)
```