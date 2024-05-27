import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def loading_data(filename: str) -> Annotated[pd.DataFrame,"input_data"]:
    """ Loads a CV File and transforms it to a Pandas DataFrame
    """
    data = pd.read_csv(filename,index_col="Date")
    return data 
