import pandas as pd
import os

def update_taxonomy():

    # 1. PATH SETUP

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "case_base_copy.csv")

    print(f"Loading case base from: {CSV_PATH}")
    
    try:
        # Force id format
        df = pd.read_csv(CSV_PATH, dtype={'id': str})
    except FileNotFoundError:
        print(f"Error: Could not finde the file {CSV_PATH}")
        return

    # 2. DEFINE MAPPING DICTIONARIES (graph_type -> value, for analytical_family and math_concept)

    family_mapping = {
        # Distribution & Dispersion
        'Histogram': 'Distribution & Dispersion',
        'Density Plot': 'Distribution & Dispersion',
        'Boxplot': 'Distribution & Dispersion',
        'Violin Plot': 'Distribution & Dispersion',
        'Swarm Plot': 'Distribution & Dispersion',
        # Relation & Correlation
        'Scatter Plot': 'Relation & Correlation',
        'Scatter with Regression': 'Relation & Correlation',
        'Bubble Chart': 'Relation & Correlation',
        'Correlation Heatmap': 'Relation & Correlation',
        # Comparison & Composition
        'Barplot': 'Comparison & Composition',
        'Pie Chart': 'Comparison & Composition',
        'Donut Chart': 'Comparison & Composition',
        'Grouped Bar Chart': 'Comparison & Composition',
        'Stacked Bar Chart': 'Comparison & Composition',
        'Treemap': 'Comparison & Composition',
        'Radar Chart': 'Comparison & Composition',
        # Time Evolution
        'Line Chart': 'Time Evolution',
        'Multiple Line Chart': 'Time Evolution',
        'Area Chart': 'Time Evolution',
        'Stacked Area Chart': 'Time Evolution',
        'Line Chart with Confidence Intervals': 'Time Evolution',
        'Candlestick Chart': 'Time Evolution',
        # Prediction Breakdown
        'SHAP Waterfall': 'Prediction Breakdown',
        'SHAP Force': 'Prediction Breakdown',
        'LIME Dashboard': 'Prediction Breakdown',
        # Feature Impact
        'SHAP Bar': 'Feature Impact',
        'LIME Bar': 'Feature Impact',
        'SHAP Beeswarm': 'Feature Impact',
        'SHAP Violin': 'Feature Impact',
        # Dependence Curve
        'ALE 1D': 'Dependence Curve',
        'ALE 2D': 'Dependence Curve',
        'SHAP Scatter': 'Dependence Curve',
        # Complex Patterns
        'SHAP Decision': 'Complex Patterns',
        'SHAP Heatmap': 'Complex Patterns'
    }

    concept_mapping = {
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

    # 3. APPLY MAPPING

    # Create the new column 'analytical_family' and update 'math_concept' using .map()
    df['analytical_family'] = df['graph_type'].map(family_mapping)
    df['math_concept'] = df['graph_type'].map(concept_mapping)

    # 4. SANITY CHECK

    # Verifying if any 'graph_type' didn't find a match in the dictionaries
    missing_families = df[df['analytical_family'].isna()]['graph_type'].unique()
    missing_concepts = df[df['math_concept'].isna()]['graph_type'].unique()
    
    if len(missing_families) > 0 or len(missing_concepts) > 0:
        print(f" WARNING: Some graph types are not mapped:")
        print(f"  - Missing in analytical_family: {missing_families}")
        print(f"  - Missing in math_concept: {missing_concepts}")
        return

    # 5. REORDER COLUMNS (Visual Hierarchy: analytical_family -> math_concept -> graph_type)

    # Extract the current columns
    cols = df.columns.tolist()
    
    # Remove 'analytical_family' from the end
    cols.remove('analytical_family')

    # Remove 'math_concept' and 'graph_type' from their initial positions
    cols.remove('math_concept')
    cols.remove('graph_type')
    
    # Search for the position of'graph_category' to insert these columns just after
    insert_idx = cols.index('graph_category') + 1
    
    # Reconstruct the columns structure following the order: ..., graph_category, analytical_family, math_concept, graph_type, ...
    cols.insert(insert_idx, 'analytical_family')
    cols.insert(insert_idx + 1, 'math_concept')
    cols.insert(insert_idx + 2, 'graph_type')
    
    df = df[cols]

    # 6. SAVE CHANGES

    df['id'] = df['id'].astype(str).str.zfill(6)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"Taxonomy updated correctly! Changes saved to: {CSV_PATH}")

if __name__ == "__main__":
    update_taxonomy()