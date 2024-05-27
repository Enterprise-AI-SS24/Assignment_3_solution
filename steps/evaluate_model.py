import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
""" DESCRIPTION:
In this Python file, you should define the step evaluate_model (use the same name). The functions require the following input parameters:
- model: The trained model that you want to evaluate. It is a ClassifierMixin object.
- X_test: The testing dataset. It is a pandas DataFrame.
- y_test: The testing labels. It is a pandas Series.
"""

@step
def evaluate_model(model:ClassifierMixin,X_test:pd.DataFrame,y_test:pd.DataFrame) -> Annotated[float,"Accuracy"]:
    """
    Evaluates the accuracy of a trained model using the testing dataset.
    """
    score = model.score(X_test,y_test)
    return score 
