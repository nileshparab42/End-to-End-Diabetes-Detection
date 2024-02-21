import pandas as pd
import os
from src.mlProject import logger
from sklearn.linear_model import LogisticRegression
import joblib
from src.mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Convert target variables to 1-dimensional arrays
        train_y = train_y.values.ravel()
        test_y = test_y.values.ravel()

        lr = LogisticRegression(penalty=self.config.penalty,
                                C=self.config.C,
                                solver=self.config.solver,
                                max_iter=self.config.max_iter,
                                class_weight=self.config.class_weight,
                                multi_class=self.config.multi_class)

        lr.fit(train_x, train_y)

        # Save the trained model
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
