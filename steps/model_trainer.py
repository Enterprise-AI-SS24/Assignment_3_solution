import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
from sklearn.base import ClassifierMixin
@step
def model_trainer(X_train: pd.DataFrame, y_train: pd.Series,best_parameters:dict)-> Tuple[Annotated[ClassifierMixin,"Model"],Annotated[float,"In_Sample_Accuracy"]]:
    """
    Trains a logistic regression model using the provided training data and computes the in-sample accuracy.
    """
    model = DecisionTreeClassifier(**best_parameters)
    model.fit(X_train,y_train)
    in_sample_score = model.score(X_train,y_train)
    return model,in_sample_score 