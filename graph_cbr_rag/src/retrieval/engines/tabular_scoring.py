import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.schemas import QueryContext
from src.retrieval.manager import CaseBaseManager
from src.retrieval.config import (
    TABULAR_WEIGHTS, 
    TOP_N_VISUAL_CANDIDATES
)
from .base import BaseEngine

class TabularScoringEngine(BaseEngine):
    """
    Computes the tabular similarity between the user's query and the candidate cases. Sorts and truncates to the Top N.
    """
    def __init__(self, manager: CaseBaseManager):
        super().__init__()
        self.manager = manager
        self.log("Loading MiniLM for dynamic text embeddings...")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _jaccard_sim(self, str1: str, str2: str) -> float:
        """Calculates Jaccard similarity between two comma-separated variable strings."""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
            
        set1 = set([x.strip().lower() for x in str1.split(',')])
        set2 = set([x.strip().lower() for x in str2.split(',')])
        
        if not set1 or not set2:
            return 0.0
            
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def execute(self, df: pd.DataFrame, query: QueryContext) -> pd.DataFrame:

        # 1. Beforehand, we prepare and cache dynamic text vectors for the query
        q_domain_emb = self.text_model.encode(query.domain, normalize_embeddings=True)
        q_task_emb = self.text_model.encode(query.analytical_task, normalize_embeddings=True)
        # Save to the engine instance so the Orchestrator can use them later
        self.latest_domain_emb = q_domain_emb
        self.latest_task_emb = q_task_emb

        if df.empty:
            self.log("Received empty DataFrame. Skipping scoring.")
            return df

        self.log(f"Scoring {len(df)} candidate tabular profiles...")

        # 2. Fetch the pre-computed matrices from the manager
        db_domain_matrix = self.manager.get_vectors_batch(df['domain_emb_path'].tolist())
        db_task_matrix = self.manager.get_vectors_batch(df['task_emb_path'].tolist())

        # 3. Vectorized Text Scoring
        domain_raw = np.dot(db_domain_matrix, q_domain_emb)
        task_raw = np.dot(db_task_matrix, q_task_emb)
        # Shift Cosine Similarity output [-1, 1] to strict [0, 1] scale (using min-max normalization for cosine similarity)
        domain_scores = (domain_raw + 1) / 2
        task_scores = (task_raw + 1) / 2

        tabular_scores = []
        
        # 4. Iterate through rows for scalar & hierarchical math
        for idx, (_, row) in enumerate(df.iterrows()):
            
            # Analytical Task and Domain scores (cosine similarity of text embeddings)
            s_domain = domain_scores[idx] * TABULAR_WEIGHTS["domain"]
            s_task = task_scores[idx] * TABULAR_WEIGHTS["analytical_task"]
            
            # Variables score (Jaccard similarity)
            val_var = self._jaccard_sim(query.variables, row.get('variables', ''))
            s_var = val_var * TABULAR_WEIGHTS["variables"]
            
            # Analytical Family score (categorical match)
            val_family = 1.0 if query.analytical_family == row.get('analytical_family', '') else 0.0
            s_family = val_family * TABULAR_WEIGHTS["analytical_family"]

            # Math Concept score (categorical match)
            val_concept = 1.0 if query.math_concept == row.get('math_concept', '') else 0.0
            s_concept = val_concept * TABULAR_WEIGHTS["math_concept"]

            # Graph Type score (categorical match)
            val_type = 1.0 if query.graph_type == row.get('graph_type', '') else 0.0
            s_type = val_type * TABULAR_WEIGHTS["graph_type"]

            # Because weights sum to 1.0, the sum of the weighted scores is the final S_tabular
            total_score = s_domain + s_task + s_var + s_family + s_concept + s_type 
            tabular_scores.append(total_score)

        # 5. Append scores and sort descending by tabular_score
        df['tabular_score'] = tabular_scores
        df_sorted = df.sort_values(by='tabular_score', ascending=False)
        
        # 6. Truncate to the Top N (defined in config.py) most similar cases according to tabular_score
        top_n = df_sorted.head(TOP_N_VISUAL_CANDIDATES).copy()
        
        self.log(f"Tabular scoring complete. Passing top {len(top_n)} candidates to visual engine.")
        return top_n