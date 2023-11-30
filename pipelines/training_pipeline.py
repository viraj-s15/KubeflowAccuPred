import kfp 
from kfp import dsl
from components import step_data_preprocessing,step_data_splitting,step_model_testing,step_hyperparam_optim

@dsl.pipeline(
    name='Customer Frequency Training Pipeline',
    description='A kubernetes pipeline for training a customer freq prediction model'
)
def customer_freq_training_pipeline():
    data_preprocessing_task = step_data_preprocessing()
    data_splitting_task = step_data_splitting().after(data_preprocessing_task)
    hyperparam_optim_task = step_hyperparam_optim().after(data_splitting_task)
    model_testing_task = step_model_testing().after(hyperparam_optim_task)
    
    data_preprocessing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    data_splitting_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    hyperparam_optim_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_testing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    kfp.compiler.Compiler().compile(
        pipeline_func=customer_freq_training_pipeline,
        package_path='compiled_pipelines/Customer_freq_training.yaml')