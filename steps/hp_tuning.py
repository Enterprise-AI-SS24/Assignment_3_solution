from zenml import step
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd
from functools import partial
from typing_extensions import Annotated

"""DESCRIPTION:
In this Python file, you can find two functions. The first one is the objective function. We are using this function to optimize our hyperparameters. 
The new tool, "Optuna", uses this function to find the best hyperparameters. Therefore, it creates a study and executes the objective function n_trials times. 
The second function is the hp_tuning function. This step function is used in our training_pipeline and returns the best hyperparameters that Optuna found. 
You can recognize that it is a step of our pipeline because the decorator "@step" is placed above it.
More about Optuna: https://optuna.readthedocs.io/en/stable/index.html
"""


def objective(trial:optuna.trial.Trial,X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series)->float:
       # Define the hyperparameters to tune
       max_depth = trial.suggest_int('max_depth', 1, 30)
       min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
       min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
       criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
       # Define the model 

       model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
       )

       model.fit(X_train, y_train)

       accuracy = accuracy_score(y_test, model.predict(X_test))
       return accuracy

@step
def hp_tuning(X_train: pd.DataFrame,X_test: pd.DataFrame, y_train: pd.Series,y_test: pd.DataFrame,trials:int=100)-> Annotated[dict,"Best hyperparameters"]:
   """
   This step tunes the hyperparameters of a logistic regression model using Optuna.
   """
   obj = partial(objective,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
   # Create a study
   study = optuna.create_study(direction="maximize")

   study.optimize(obj, n_trials=trials)
   # Get the best hyperparameters
   best_params = study.best_params
   # Return the best hyperparameters
   return best_params