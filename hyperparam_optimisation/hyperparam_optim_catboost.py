def hyperparam_optim_catboost() -> None:
    import numpy as np
    from catboost import Pool, CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from aim import Run
    import optuna
    import logging
    import os
    import json
    import pickle

    logging.basicConfig(level=logging.INFO)
    logging.info("Started Hyperparameter Optimization")

    try:
        X_train = np.load('data/splits/X_train.npy')
        y_train = np.load('data/splits/y_train.npy')

        def objective(trial):
            params = {
                'loss_function': 'RMSE',
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'border_count': trial.suggest_int('border_count', 5, 255),
                'thread_count': -1,
                'verbose': False,
            }

            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            train_pool = Pool(X_train_split, label=y_train_split)
            val_pool = Pool(X_val, label=y_val)

            logging.info("Started training model")
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, verbose_eval=False)

            y_pred = model.predict(val_pool)
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

        logging.info("Training model with best hyperparams")
        best_model = CatBoostRegressor(**best_params)
        best_model.fit(Pool(X_train, label=y_train))

        logging.info("Saving model into the model folder")
        model_directory = 'model'
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        model_filename = os.path.join(model_directory, 'best_catboost_model.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)

        logging.info('#' * 100)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    hyperparam_optim_catboost()

