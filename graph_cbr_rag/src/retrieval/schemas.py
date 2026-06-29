from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class QueryContext:
    """
    The input structured type for the Retrieval Module. 
    Represents a new user query to be matched against the Case Base.
    """
    img_path: str # Not the "000000.png" path, but where the input image is stored locally
    domain: str
    graph_category : str
    graph_type: str
    analytical_task: str
    variables: str
    
    # field(init=False) because we don't pass this when creating the object; 
    # it is calculated automatically inside __post_init__
    analytical_family: str = field(init=False) 
    math_concept: str = field(init=False)

    def __post_init__(self):
        """
        Automatically executed immediately after the object is created.
        This isolates the logic for calculating the analytical family and the math concept.
        """
        # Create the key matching the logic defined in update_taxonomy.py
        mapping_key = self.graph_type
        
        # Dictionary containing the graph types mapped to their analytical families
        family_map = {
            # Family 1: Distribution & Dispersion
            'Histogram': 'Distribution & Dispersion',
            'Density Plot': 'Distribution & Dispersion',
            'Boxplot': 'Distribution & Dispersion',
            'Violin Plot': 'Distribution & Dispersion',
            'Swarm Plot': 'Distribution & Dispersion',
            # Family 2: Relation & Correlation
            'Scatter Plot': 'Relation & Correlation',
            'Scatter with Regression': 'Relation & Correlation',
            'Bubble Chart': 'Relation & Correlation',
            'Correlation Heatmap': 'Relation & Correlation',
            # Family 3: Comparison & Composition
            'Barplot': 'Comparison & Composition',
            'Pie Chart': 'Comparison & Composition',
            'Donut Chart': 'Comparison & Composition',
            'Grouped Bar Chart': 'Comparison & Composition',
            'Stacked Bar Chart': 'Comparison & Composition',
            'Treemap': 'Comparison & Composition',
            'Radar Chart': 'Comparison & Composition',
            # Family 4: Time Evolution
            'Line Chart': 'Time Evolution',
            'Multiple Line Chart': 'Time Evolution',
            'Area Chart': 'Time Evolution',
            'Stacked Area Chart': 'Time Evolution',
            'Line Chart with Confidence Intervals': 'Time Evolution',
            'Candlestick Chart': 'Time Evolution',
            # Family 5: Prediction Breakdown
            'SHAP Waterfall': 'Prediction Breakdown',
            'SHAP Force': 'Prediction Breakdown',
            'LIME Dashboard': 'Prediction Breakdown',
            # Family 6: Feature Impact
            'SHAP Bar': 'Feature Impact',
            'LIME Bar': 'Feature Impact',
            'SHAP Beeswarm': 'Feature Impact',
            'SHAP Violin': 'Feature Impact',
            # Family 7: Dependence Curve
            'ALE 1D': 'Dependence Curve',
            'ALE 2D': 'Dependence Curve',
            'SHAP Scatter': 'Dependence Curve',
            # Family 8: Complex Patterns
            'SHAP Decision': 'Complex Patterns',
            'SHAP Heatmap': 'Complex Patterns'
        }

        # Dictionary containing the graph types mapped to their math concepts
        concept_map = {
            # Distribution & Dispersion
            'Histogram': 'Frequency & Probability Density',
            'Density Plot': 'Frequency & Probability Density',
            'Boxplot': 'Dispersion & Outliers',
            'Violin Plot': 'Dispersion & Outliers',
            'Swarm Plot': 'Dispersion & Outliers',
            # Relation & Correlation
            'Scatter Plot': 'Multivariate Correlation & Trend',
            'Scatter with Regression': 'Multivariate Correlation & Trend',
            'Bubble Chart': 'Multivariate Correlation & Trend',
            'Correlation Heatmap': 'Correlation Matrix',
            # Comparison & Composition
            'Barplot': 'Univariate Composition & Magnitude',
            'Pie Chart': 'Univariate Composition & Magnitude',
            'Donut Chart': 'Univariate Composition & Magnitude',
            'Grouped Bar Chart': 'Multivariate & Hierarchical Composition',
            'Stacked Bar Chart': 'Multivariate & Hierarchical Composition',
            'Treemap': 'Multivariate & Hierarchical Composition',
            'Radar Chart': 'Multivariate Profile',
            # Time Evolution
            'Line Chart': 'Single Temporal Trend & Volume',
            'Area Chart': 'Single Temporal Trend & Volume',
            'Line Chart with Confidence Intervals': 'Single Temporal Trend & Volume',
            'Multiple Line Chart': 'Comparative & Cumulative Temporal Trend',
            'Stacked Area Chart': 'Comparative & Cumulative Temporal Trend',
            'Candlestick Chart': 'Financial Price Action',
            # Prediction Breakdown
            'SHAP Waterfall': 'Additive Marginal Contribution',
            'SHAP Force': 'Additive Marginal Contribution',
            'LIME Dashboard': 'Local Weight Approximation',
            # Feature Impact
            'SHAP Bar': 'Feature Weight Ranking',
            'LIME Bar': 'Feature Weight Ranking',
            'SHAP Beeswarm': 'Feature Impact Distribution',
            'SHAP Violin': 'Feature Impact Distribution',
            # Dependence Curve
            'ALE 1D': '1D Marginal Effect',
            'SHAP Scatter': '1D Marginal Effect',
            'ALE 2D': 'Bivariate Feature Interaction',
            # Complex Patterns
            'SHAP Decision': 'Cumulative Cohort Trajectory',
            'SHAP Heatmap': 'Population-Level Impact Matrix'
        }
        
        # Automatically assign the family and concept, defaulting to Unknown to catch mapping errors
        self.analytical_family = family_map.get(mapping_key, "Unknown Family")
        self.math_concept = concept_map.get(mapping_key, "Unknown Concept") 


@dataclass
class RetrievedCase:
    """
    The output structured type for the Retrieval Orchestrator.
    Contains the final scores and the complete original context.
    """
    case_id: str
    final_score: float
    tabular_score: float
    visual_score: float
    
    # This dictionary will hold the entire row from the CSV
    metadata: Dict[str, Any]