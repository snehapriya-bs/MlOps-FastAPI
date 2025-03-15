import sys
from pathlib import Path
import pdb 

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_rental_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )

    # Pipeline fitting
    bikeshare_rental_pipe.fit(X_train,y_train)
    y_pred = bikeshare_rental_pipe.predict(X_test)

    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_rental_pipe)

    # printing the score
    print("R2 Score (in %):", r2_score(y_test, y_pred)*100)
    
if __name__ == "__main__":
    print("Running")
    run_training()
