def model_testing() -> None:
    import numpy as np
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    import pickle
    from aim import Run
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("Started Model Testing")

    try:
        X_test = np.load("data/splits/X_test.npy")
        y_test = np.load("data/splits/y_test.npy")

        model_directory = "model"
        model_filename = os.path.join(model_directory, "best_catboost_model.pkl")

        with open(model_filename, "rb") as file:
            best_catboost = pickle.load(file)

        y_pred = best_catboost.predict(X_test)

        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Test RMSE: {test_rmse}")

        run = Run()
        run.track(test_rmse, name="Test_RMSE", context={"subset": "test"})

        logging.info("Model testing has completed")
        logging.info("Model Training pipeline has ended")
        logging.info("#" * 100)
    except Exception as e:
        print(f"An error occurred during model testing: {e}")
