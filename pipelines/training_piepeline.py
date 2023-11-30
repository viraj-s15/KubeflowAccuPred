import kfp 
from kfp import dsl
from components import step_data_preprocessing,step_data_splitting,step_model_testing,step_hyperparam_optimisation

@dsl.pipeline(
    name='Customer Frequency Training Pipeline',
    description=''
)
def customer_freq_training_pipeline(data_path: str):
    vop = dsl.VolumeOp(
        name="t-vol",
        resource_name="t-vol", 
        size="1Gi", 
        modes=dsl.VOLUME_MODE_RWO
    )
    
    data_preprocessing_task = step_data_preprocessing().add_pvolumes({data_path: vop.volume})
    data_splitting_task = step_data_splitting().add_pvolumes({data_path: vop.volume}).after(data_preprocessing_task)
    hyperparam_optim_task = step_hyperparam_optimisation().add_pvolumes({data_path: vop.volume}).after(data_splitting_task)
    model_testing_task = step_model_testing().add_pvolumes({data_path: vop.volume}).after(hyperparam_optim_task)
    
    data_preprocessing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    data_splitting_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    hyperparam_optim_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_testing_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

pipeline_func = customer_freq_training_pipeline

kfp.compiler.Compiler().compile(
    pipeline_func=customer_freq_training_pipeline,
    package_path='Customer_freq_training.yaml')
