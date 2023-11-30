from kfp import Client
from pipelines import customer_freq_training_pipeline
import datetime

client = Client()

DATA_PATH = 'data/'

pipeline_func = customer_freq_training_pipeline

experiment_name = 'customer_freq_prediction' + str(datetime.datetime.now().date())
run_name = pipeline_func.__name__ + ' run'

arguments = {"data_path": DATA_PATH}

run_result = client.create_run_from_pipeline_func(
    pipeline_func=pipeline_func,
    experiment_name=experiment_name,
    run_name=run_name,
    namespace="kubeflow",
    arguments=arguments
)
