from kfp import Client
import datetime

client = Client()

DATA_PATH = 'data/'


experiment_name = 'customer_freq_prediction' + str(datetime.datetime.now().date())
run_name = 'customer_freq_prediction' + ' run'

run_result = client.create_run_from_pipeline_package(
    pipeline_func='compiled_pipelines/Customer_freq_training.yaml',
    experiment_name=experiment_name,
    run_name=run_name,
    namespace="kubeflow",
)
