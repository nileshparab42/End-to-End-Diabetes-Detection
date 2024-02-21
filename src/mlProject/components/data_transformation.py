import os
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # Identify the indices of the independent variables
        independent_cols = data.columns[:-1]

        # Apply StandardScaler only to the independent variables
        scaler = StandardScaler()
        train[independent_cols] = scaler.fit_transform(train[independent_cols])
        test[independent_cols] = scaler.transform(test[independent_cols])

        # Reshape the target variable to 1-dimensional array
        y_train = train.iloc[:, -1].values.ravel()
        y_test = test.iloc[:, -1].values.ravel()

        # Save the transformed datasets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
