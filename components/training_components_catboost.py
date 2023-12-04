from kfp import dsl
from data_preprocessing import data_preprocessing
from data_splits import data_spltting
from hyperparam_optimisation_catboost import hyperparam_optimisation_catboost
from model_testing import model_testing


@dsl.component(
    packages_to_install=[
        "pandas==2.1.3",
        "numpy==1.26.2",
        "logging==0.4.9.6",
        "scikit-learn==1.3.2",
    ],
    base_image="python:3.11",
)
def step_data_preprocessing():
    data_preprocessing()


@dsl.component(
    packages_to_install=[
        "pandas==2.1.3",
        "numpy==1.26.2",
        "logging==0.4.9.6",
        "scikit-learn==1.3.2",
    ],
    base_image="python:3.11",
)
def step_data_splitting():
    data_spltting()


@dsl.component(
    packages_to_install=[
        "numpy==1.26.2",
        "logging==0.4.9.6",
        "scikit-learn==1.3.2",
        "aim==3.17.5",
        "optuna==3.4.0",
        "catboost==1.2.2",
    ],
    base_image="python:3.11",
)
def step_hyperparam_optim():
    hyperparam_optimisation_catboost()


@dsl.component(
    packages_to_install=[
        "numpy==1.26.2",
        "logging==0.4.9.6",
        "scikit-learn==1.3.2",
        "aim==3.17.5",
        "xgboost==2.0.2",
    ],
    base_image="python:3.11",
)
def step_model_testing():
    model_testing
