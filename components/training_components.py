import kfp 
from data_preprocessing import data_preprocessing
from data_splits import data_spltting  
from hyperparam_optimisation import hyperparam_optim
from model_testing import model_testing

step_data_preprocessing = kfp.components.create_component_from_func(
    func=data_preprocessing,
    base_image='python:3.11',
    packages_to_install=['pandas==2.1.3','numpy==1.26.2','logging==0.4.9.6','scikit-learn==1.3.2']
)

step_data_splitting = kfp.components.create_component_from_func(
    func=data_spltting,
    base_image='python:3.11',
    packages_to_install=['pandas==2.1.3','numpy==1.26.2','logging==0.4.9.6','scikit-learn==1.3.2']
)

step_hyperparam_optimisation = kfp.components.create_component_from_func(
    func=hyperparam_optim,
    base_image='python:3.11',
    packages_to_install=['numpy==1.26.2','logging==0.4.9.6','scikit-learn==1.3.2','aim==3.17.5','optuna==3.4.0','xgboost==2.0.2']
)

step_model_testing = kfp.components.create_component_from_func(
    func=model_testing,
    base_image='python:3.11',
    packages_to_install=['numpy==1.26.2','logging==0.4.9.6','scikit-learn==1.3.2','aim==3.17.5','xgboost==2.0.2']
)