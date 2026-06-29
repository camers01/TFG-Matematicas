"""
Configuration parameters and weights for the Retrieval Module.
Centralizing these values allows for easy tuning without touching engine logic.
"""

# 1. ORCHESTRATION WEIGHTS

FUSION_WEIGHT_Z = 0.65
TOP_N_VISUAL_CANDIDATES = 20
TOP_K_FINAL_RESULTS = 3


# 2. TABULAR SCORING WEIGHTS

TABULAR_WEIGHTS = {
    # HIGH (60%)
    "analytical_family": 0.20, 
    "math_concept": 0.20,
    "graph_type": 0.20,
    # MEDIUM (30%)
    "analytical_task": 0.20,            
    "domain": 0.10,         
    # LOW (10%)
    "variables": 0.10      
}