def data_spltting() -> None:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os
    import logging
    logging.basicConfig(level = logging.INFO)

    logging.info("Started Data Splitting Component")
    
    processed_data = pd.read_csv("data/processed_data.csv")
    logging.info("Reading data from data/processed_data.csv")
    target_column = 'Frequency of Purchases'
    X = processed_data.loc[:, processed_data.columns != target_column]
    y = processed_data.loc[:, processed_data.columns == target_column]
    directory = 'data/splits'

    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory '{directory}' created successfully!")
    else:
        logging.info(f"Directory '{directory}' already exists. Skipping creation")    
        logging.warn(f"This behaviour is due to cache, make sure this is what you intended")    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=1234)
        
    np.save(f'data/splits/X_train.npy', X_train)
    np.save(f'data/splits/X_test.npy', X_test)
    np.save(f'data/splits/y_train.npy', y_train)
    np.save(f'data/splits/y_test.npy', y_test)
    logging.info("Splits have been saved into the `data/splits/` directory")
    print('#' * 100)
    
if __name__ == "__main__":
    data_spltting()