def hyperparam_optim() -> None:
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from aim import Run
    import optuna
    import logging
    import os
    import json
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Started Hyperparameter Optimization")

    try:
        X_train = np.load('data/splits/X_train.npy')
        y_train = np.load('data/splits/y_train.npy')

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'eta': trial.suggest_float('eta', 1e-3, 0.5),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }

            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
            dval = xgb.DMatrix(X_val, label=y_val)
            logging.info("Started training model")
            bst = xgb.train(params, dtrain, num_boost_round=trial.suggest_int('n_estimators', 100, 1000, step=100), evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)

            y_pred = bst.predict(dval)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            run = Run()
            run["hparams"] = {k: v for k, v in params.items() if k != 'eval_metric'}
            run.track(rmse, name='RMSE', context={"subset": "validation"})

            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")

        folder_path = 'params'
        if not os.path.exists(folder_path):
            logging.info("Params folder does not exist, creating it")
            os.makedirs(folder_path)
        else:
            logging.info("Params folder already exists, skipping creation")

        run_num = 1
        file_path = os.path.join(folder_path, f'run_{run_num}.json')

        while os.path.exists(file_path):
            run_num += 1
            file_path = os.path.join(folder_path, f'run_{run_num}.json')

        with open(file_path, 'w') as file:
            json.dump(best_params, file)

        logging.info("Completed Hyperparameter Optimization")
        logging.info(f"Best hyperparameters saved to: {file_path}")
        logging.info('#' * 100)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    hyperparam_optim()