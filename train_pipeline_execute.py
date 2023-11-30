import argparse
from kfp import Client
import datetime

def run_pipeline(host):
    client = Client(host=host)
    DATA_PATH = 'data/'

    experiment_name = 'customer_freq_prediction' + str(datetime.datetime.now().date())
    run_name = 'customer_freq_prediction' + ' run'

    run_result = client.create_run_from_pipeline_package(
        pipeline_func='compiled_pipelines/Customer_freq_training.yaml',
        experiment_name=experiment_name,
        run_name=run_name,
        namespace="kubeflow",
        enable_caching=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KFP pipeline")
    parser.add_argument("--host", type=str, required=True, help="Kubeflow Pipelines host URL")
    args = parser.parse_args()

    run_pipeline(args.host)
