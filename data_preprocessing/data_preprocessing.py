def data_preprocessing() -> None:
    import pandas as pd
    import numpy as np
    import logging
    from sklearn.preprocessing import LabelEncoder
    
    logging.basicConfig(level=logging.INFO)
    print('#' * 100)
    logging.info("Starting Data Preprocessing Component")
    
    try:
        filename = "data/data.csv"
        df = pd.read_csv(filename)
        logging.info("File has been loaded")

        missing = sum(df.isna().sum())
        logging.info(f"Missing elements: {missing}")

        cols = df.columns
        object_cols = [col for col in cols if df[col].dtype == "object"]

        if object_cols:
            logging.info(f"The columns which contain objects are: {', '.join(object_cols)}")
            logging.info("These columns will be encoded")
            le = LabelEncoder()
            logging.info("Encoding.......")
            for col in object_cols:
                df[col] = le.fit_transform(df[col])
            logging.info("Encoding complete")

        df.to_csv("data/processed_data.csv", index=False)
        logging.info("Preprocessed csv file has been saved into data/processed_data.csv")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    print('#' * 100)


if __name__ == "__main__":
    data_preprocessing()
