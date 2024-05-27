from steps import hp_tuning,model_trainer,evaluate_model,split_data,loading_data
from zenml import pipeline

@pipeline
def training_pipeline():
    """
    Executes a full training pipeline on weather data to predict rain tomorrow.
    """
    dataset = loading_data("./data/Weather_Perth_transformed.csv")
    X_train,X_test,y_train,y_test = split_data(dataset,"RainTomorrow")

    best_parameters = hp_tuning(X_train,X_test,y_train,y_test)
    model,in_sample_score = model_trainer(X_train,y_train,best_parameters)
    out_of_sample_score = evaluate_model(model,X_test,y_test)
 