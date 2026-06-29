import pandas as pd
from src.retrieval.schemas import QueryContext
from .base import BaseEngine

class StrictFilterEngine(BaseEngine):
    """
    Applies hard boolean masks to the database based on 
    the selected columns of the user's query.
    """
    def execute(self, df: pd.DataFrame, query: QueryContext) -> pd.DataFrame:

        self.log(f"Received {len(df)} total cases in database.")
        
        mask = (
            (df['graph_category'] == query.graph_category)
        )

        filtered_df = df[mask].copy()
        
        self.log(f"Cases remaining after Strict Filtering: {len(filtered_df)}")

        return filtered_df.copy()