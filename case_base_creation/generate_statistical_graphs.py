import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import squarify            
import mplfinance as mpf
import yfinance as yf
from math import pi

# =====================================================================
# 0. AUXILIARY FUNCTIONS (Establishes the 15 columns for each row and saves the image)
# =====================================================================

def _build_row(current_id, domain, graph_type, math_concept, analytical_task, variables):
    str_id = f"{current_id:06d}"
    return {
        'id': str_id,
        'img_path': f"images/{str_id}.png",
        'domain_emb_path': f"embeddings_domain/{str_id}.npy",
        'task_emb_path': f"embeddings_task/{str_id}.npy",
        'visual_emb_path': f"embeddings_visual/{str_id}.npy",
        'domain': domain,
        'graph_category': "statistical",
        'graph_type': graph_type,
        'math_concept': math_concept,
        'analytical_task': analytical_task,
        'variables': variables,
        'solution_insights': "",
        'qwen_insight': "",
        'pixtral_insight': "",
        'idefics_insight': ""
    }

def _save_plot(current_id):
    """Saves the current image in the folder images/ and closes the plot."""
    os.makedirs("images", exist_ok=True)
    str_id = f"{current_id:06d}"
    plt.tight_layout()
    plt.savefig(f"images/{str_id}.png", dpi=150)
    plt.close()

# =====================================================================
# 1. DISTRIBUTION GRAPHS
# =====================================================================

def generate_histogram(df, col, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, kde=False)
    plt.title(f"Histogram of {col}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Histogram", "Distribution", f"Frequency distribution analysis of {col}", col)

def generate_kde(df, col, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x=col, fill=True)
    plt.title(f"Density Plot of {col}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Density Plot", "Probability Density", f"Density estimation of {col}", col)

def generate_boxplot(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=cat, y=num)
    plt.title(f"Boxplot of {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Boxplot", "Dispersion and Outliers", f"Dispersion and central tendency of {num} grouped by {cat}", f"{cat}, {num}")

def generate_violin(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x=cat, y=num)
    plt.title(f"Violin Plot of {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Violin Plot", "Distribution Density", f"Density distribution comparison of {num} across {cat}", f"{cat}, {num}")

def generate_swarm(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x=cat, y=num, size=1.5)
    plt.title(f"Swarm Plot of {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Swarm Plot", "Data Point Distribution", f"Individual observation distribution of {num} across {cat}", f"{cat}, {num}")

# =====================================================================
# 2. RELATION AND CORRELATION GRAPHS
# =====================================================================

def generate_scatter(df, x, y, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Scatter Plot: {y} vs {x}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Scatter Plot", "Correlation", f"Correlation analysis between {x} and {y}", f"{x}, {y}")

def generate_scatter_reg(df, x, y, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=x, y=y)
    plt.title(f"Regression: {y} vs {x}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Scatter with Regression", "Linear Trend", f"Linear regression trend between {x} and {y}", f"{x}, {y}")

def generate_bubble(df, x, y, size, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, size=size, sizes=(20, 400), alpha=0.6)
    plt.title(f"Bubble Chart: {y} vs {x} (Size: {size})")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Bubble Chart", "Multivariate Correlation", f"Trivariate relationship between {x}, {y} and magnitude of {size}", f"{x}, {y}, {size}")

def generate_heatmap(df, cols, domain, current_id):
    plt.figure(figsize=(8, 6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Correlation Heatmap", "Correlation Matrix", f"Cross-correlation analysis among multiple numerical variables", ", ".join(cols))

# =====================================================================
# 3. COMPARISON, MAGNITUDE AND COMPOSITION GRAPHS
# =====================================================================

def generate_barplot(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x=cat, y=num, errorbar=None)
    plt.title(f"Average {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Barplot", "Magnitude Comparison", f"Comparison of average {num} across {cat}", f"{cat}, {num}")

def generate_pie(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    data_agg = df.groupby(cat)[num].sum()
    plt.pie(data_agg, labels=data_agg.index, autopct='%1.1f%%')
    plt.title(f"Pie Chart: Proportion of {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Pie Chart", "Proportional Composition", f"Proportional distribution of total {num} across {cat}", f"{cat}, {num}")

def generate_donut(df, cat, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    data_agg = df.groupby(cat)[num].sum()
    plt.pie(data_agg, labels=data_agg.index, autopct='%1.1f%%', wedgeprops=dict(width=0.4))
    plt.title(f"Donut Chart: Proportion of {num} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Donut Chart", "Proportional Composition", f"Proportional distribution of total {num} across {cat}", f"{cat}, {num}")

def generate_grouped_bar(df, cat1, cat2, num, domain, current_id):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x=cat1, y=num, hue=cat2, errorbar=None)
    plt.title(f"{num} by {cat1} and {cat2}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Grouped Bar Chart", "Multivariate Comparison", f"Comparison of {num} grouped by {cat1} and sub-grouped by {cat2}", f"{cat1}, {cat2}, {num}")

def generate_stacked_bar(df, cat1, cat2, num, domain, current_id):
    plt.close('all')
    plt.figure(figsize=(8, 6))
    data_pivot = df.groupby([cat1, cat2])[num].sum().unstack()
    ax = data_pivot.plot(kind='bar', stacked=True, figsize=(8,6))
    plt.title(f"Stacked {num} by {cat1} and {cat2}")
    _save_plot(current_id)
    plt.close(ax.figure) 
    return _build_row(current_id, domain, "Stacked Bar Chart", "Cumulative Composition", f"Cumulative composition of {num} across {cat1} divided by {cat2}", f"{cat1}, {cat2}, {num}")

def generate_treemap(df, cat1, cat2, num, domain, current_id):
    plt.figure(figsize=(10, 8))
    # Group by both categories to calculate sizes
    data_agg = df.groupby([cat1, cat2])[num].sum().reset_index()
    # Remove rows where the numerical value is 0 or negative, as Squarify throws a ZeroDivisionError if it attempts to plot a rectangle of size 0.
    data_agg = data_agg[data_agg[num] > 0]
    # Create combined labels (Example: "Europe - Spain")
    labels = data_agg.apply(lambda x: f"{x[cat1]}\n{x[cat2]}", axis=1)
    squarify.plot(sizes=data_agg[num], label=labels, alpha=0.8)
    plt.title(f"Treemap of {num} grouped by {cat1} and {cat2}")
    plt.axis('off') # Treemap does not need axes
    _save_plot(current_id)
    return _build_row(current_id, domain, "Treemap", "Hierarchical Proportions", f"Hierarchical proportional composition of {num} segmented by {cat1} and {cat2}", f"{cat1}, {cat2}, {num}")

def generate_radar(df, cat_entity, num_cols, domain, current_id):
    # Calculate the mean of the metrics for each entity
    data_agg = df.groupby(cat_entity)[num_cols].mean().reset_index()
    # Reproducible random sampling of up to 5 categories for visual clarity
    data_agg = data_agg.sample(frac=1, random_state=23).head(5)
    # MIN-MAX NORMALIZATION: normalize each column from 0 to 1 to avoid scale issues in the radar graph
    for col in num_cols:
        min_val = data_agg[col].min()
        max_val = data_agg[col].max()
        if max_val > min_val:
            data_agg[col] = (data_agg[col] - min_val) / (max_val - min_val)
        else:
            data_agg[col] = 0.5 # Valor por defecto si todos los datos son idénticos
    # Preparing the polar graph
    categories = num_cols
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the polygon
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    # We draw a polygon for each row (entity) 
    for i, row in data_agg.iterrows():
        values = row[num_cols].values.flatten().tolist()
        values += values[:1] # Close the polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row[cat_entity])
        ax.fill(angles, values, alpha=0.1)
    # Fix the limits from 0 to 1 (because of the normalization) and we hide the radial numbers 
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f"Radar Chart profile by {cat_entity}\n(Normalized 0-1)")
    _save_plot(current_id)
    # We format the columns as a string separated by commas
    vars_str = f"{cat_entity}, " + ", ".join(num_cols)
    return _build_row(current_id, domain, "Radar Chart", "Multivariate Profile", f"Multivariate profile and comparison of {len(num_cols)} numerical attributes grouped by {cat_entity} (normalized scales)", vars_str)

# =====================================================================
# 4. EVOLUTION AND TENDENCY GRAPHS
# =====================================================================

def generate_line(df, time_col, num, domain, current_id):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=time_col, y=num, errorbar=None)
    plt.title(f"Evolution of {num} over {time_col}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Line Chart", "Temporal Trend", f"Temporal evolution analysis of {num}", f"{time_col}, {num}")

def generate_multiple_line(df, time_col, num, cat, domain, current_id):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=time_col, y=num, hue=cat, errorbar=None)
    plt.title(f"Evolution of {num} over {time_col} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Multiple Line Chart", "Comparative Temporal Trend", f"Comparative temporal evolution of {num} across {cat}", f"{time_col}, {num}, {cat}")

def generate_area(df, time_col, num, domain, current_id):
    plt.figure(figsize=(10, 5))
    data_agg = df.groupby(time_col)[num].sum().reset_index()
    plt.fill_between(data_agg[time_col], data_agg[num], alpha=0.5)
    plt.plot(data_agg[time_col], data_agg[num])
    plt.title(f"Area Chart of {num} over {time_col}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Area Chart", "Cumulative Trend", f"Volume and temporal evolution of {num}", f"{time_col}, {num}")

def generate_stacked_area(df, time_col, num, cat, domain, current_id):
    # Pivot in order to have the time in the index and the categories as columns
    data_pivot = df.pivot_table(index=time_col, columns=cat, values=num, aggfunc='sum').fillna(0)
    plt.figure(figsize=(10, 6))
    plt.stackplot(data_pivot.index, data_pivot.T, labels=data_pivot.columns, alpha=0.8)
    plt.legend(loc='upper left')
    plt.title(f"Stacked Area of {num} over {time_col} by {cat}")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Stacked Area Chart", "Cumulative Temporal Composition", f"Cumulative volume and composition evolution of {num} across {cat} over time", f"{time_col}, {num}, {cat}")

def generate_line_ci(df, time_col, num, domain, current_id):
    plt.figure(figsize=(10, 5))
    # The parameter errorbar='ci' tells Seaborn to draw the confidence interval (95% default) around the line
    sns.lineplot(data=df, x=time_col, y=num, errorbar=('ci', 95))
    plt.title(f"Evolution of {num} with 95% Confidence Intervals")
    _save_plot(current_id)
    return _build_row(current_id, domain, "Line Chart with Confidence Intervals", "Temporal Trend and Uncertainty", f"Temporal evolution and uncertainty estimation (confidence intervals) of {num}", f"{time_col}, {num}")

def generate_candlestick(df, time_col, open_c, high_c, low_c, close_c, domain, current_id):
    # mplfinance requires a very specific DataFrame: Datetime Index and columnas called Open, High, Low, Close
    df_candle = df[[time_col, open_c, high_c, low_c, close_c]].copy()
    df_candle[time_col] = pd.to_datetime(df_candle[time_col])
    df_candle.set_index(time_col, inplace=True)
    df_candle.rename(columns={open_c: 'Open', high_c: 'High', low_c: 'Low', close_c: 'Close'}, inplace=True)
    os.makedirs("images", exist_ok=True)
    str_id = f"{current_id:06d}"
    img_path = f"images/{str_id}.png"
    # mplfinance saves the image directly, we do not use plt.savefig
    mpf.plot(df_candle, type='candle', style='yahoo', title="Financial Price Action", savefig=img_path)
    return _build_row(current_id, domain, "Candlestick Chart", "Financial Price Action", f"Financial market fluctuation tracking representing open, high, low, and close prices", f"{time_col}, {open_c}, {high_c}, {low_c}, {close_c}")

# =====================================================================
# MAIN LOOP FOR GRAPH GENERATION
# =====================================================================

def main():

    ##################### LOAD AND CLEAN THE DATASETS TO BE USED #####################

    df_penguins = sns.load_dataset("penguins").dropna()
    df_iris = sns.load_dataset("iris").dropna()
    df_titanic = sns.load_dataset("titanic").dropna(subset=['age', 'fare', 'class', 'sex', 'survived', 'embark_town'])
    df_diamonds = sns.load_dataset("diamonds").dropna().sample(500, random_state=23) # Reduced sample to avoid crashing when plotting
    df_tips = sns.load_dataset("tips").dropna()
    df_mpg = sns.load_dataset("mpg").dropna()
    df_taxis = sns.load_dataset("taxis").dropna().sample(500, random_state=23) # Reduced sample to avoid crashing when plotting
    df_healthexp = sns.load_dataset("healthexp").dropna()
    df_flights = sns.load_dataset("flights").dropna()
    df_fmri = sns.load_dataset("fmri").dropna().sample(500, random_state=23) # Reduced sample to avoid crashing when plotting
    df_crashes = sns.load_dataset("car_crashes").dropna() 
    df_planets = sns.load_dataset("planets").dropna()
    df_dots = sns.load_dataset("dots").dropna()
    df_attention = sns.load_dataset("attention").dropna()
    df_dowjones = sns.load_dataset("dowjones").dropna()

    # Extra download for 4 financial assets to be used in classical time series (Lines, Areas)
    time_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    df_fin_time = {}
    for t in time_tickers:
        try:
            df_t = yf.download(t, start="2025-09-01", end="2026-01-01", progress=False).reset_index()
            if isinstance(df_t.columns, pd.MultiIndex): df_t.columns = df_t.columns.droplevel(1)
            df_t.rename(columns={'index': 'Date', 'Date': 'Date'}, inplace=True)
            df_fin_time[t] = df_t
        except Exception:
            pass

    ##################### CONFIGURATION DICTIONARY WITH THE CHOSEN DATASETS #####################
    
    generation_config = [

        # BIOLOGY (penguins, iris)

        {"dataset_name": "penguins", "domain": "Biology", "data": df_penguins,
         "targets_1D_num": ["body_mass_g", "flipper_length_mm"], 
         "targets_cat_num_dist": [("species", "body_mass_g"), ("sex", "flipper_length_mm"), ("island", "bill_length_mm")],
         "targets_2D_num": [("bill_length_mm", "bill_depth_mm"), ("flipper_length_mm", "body_mass_g"), ("bill_length_mm", "body_mass_g")],
         "targets_3D_num": [("bill_length_mm", "bill_depth_mm", "body_mass_g"), ("flipper_length_mm", "body_mass_g", "bill_length_mm")],
         "targets_multi_num": [["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], ["bill_length_mm", "flipper_length_mm", "body_mass_g"]],
         "targets_cat_num_comp": [("species", "body_mass_g"), ("island", "body_mass_g"), ("sex", "body_mass_g")],
         "targets_2cat_num": [("island", "species", "body_mass_g"), ("species", "sex", "body_mass_g"), ("island", "sex", "flipper_length_mm")],
         "targets_radar": [("species", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]), ("island", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]), ("sex", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},
         
        {"dataset_name": "iris", "domain": "Biology", "data": df_iris,
         "targets_1D_num": ["sepal_length", "petal_width"], 
         "targets_cat_num_dist": [("species", "sepal_width"), ("species", "petal_length")],
         "targets_2D_num": [("sepal_length", "petal_length"), ("sepal_width", "petal_width")],
         "targets_3D_num": [("sepal_length", "sepal_width", "petal_length"), ("sepal_width", "petal_length", "petal_width")],
         "targets_multi_num": [["sepal_length", "sepal_width", "petal_length", "petal_width"]],
         "targets_cat_num_comp": [("species", "sepal_length"), ("species", "petal_width")],
         "targets_2cat_num": [],
         "targets_radar": [("species", ["sepal_length", "sepal_width", "petal_length", "petal_width"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},

        # DEMOGRAPHICS AND ECONOMY (titanic, diamonds, tips, crashes)

        {"dataset_name": "titanic", "domain": "Demographics", "data": df_titanic,
         "targets_1D_num": ["age", "fare"], 
         "targets_cat_num_dist": [("class", "age"), ("survived", "fare"), ("sex", "age")],
         "targets_2D_num": [("age", "fare")], 
         "targets_3D_num": [], 
         "targets_multi_num": [],
         "targets_cat_num_comp": [("class", "fare"), ("survived", "age"), ("embark_town", "fare"), ("sex", "fare")],
         "targets_2cat_num": [("class", "sex", "age"), ("class", "survived", "fare"), ("embark_town", "class", "fare"), ("embark_town", "sex", "age")],
         "targets_radar": [],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},
         
        {"dataset_name": "diamonds", "domain": "Retail", "data": df_diamonds,
         "targets_1D_num": ["price", "carat", "depth"], 
         "targets_cat_num_dist": [("cut", "price"), ("color", "carat"), ("clarity", "price")],
         "targets_2D_num": [("carat", "price"), ("depth", "price"), ("table", "price"), ("x", "price")],
         "targets_3D_num": [("carat", "depth", "price"), ("table", "depth", "price"), ("x", "y", "price"), ("carat", "x", "price")],
         "targets_multi_num": [["carat", "depth", "table", "price"], ["carat", "x", "y", "z"], ["depth", "table", "price", "x", "y", "z"]],
         "targets_cat_num_comp": [("cut", "price"), ("color", "price"), ("clarity", "price")],
         "targets_2cat_num": [("cut", "color", "price"), ("cut", "clarity", "price"), ("color", "clarity", "carat")],
         "targets_radar": [("cut", ["carat", "depth", "table", "price"]), ("color", ["carat", "depth", "table", "price"]), ("clarity", ["carat", "depth", "table", "price"]), ("cut", ["x", "y", "z", "carat"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},
         
        {"dataset_name": "tips", "domain": "Hospitality & Services", "data": df_tips,
         "targets_1D_num": ["total_bill", "tip"], 
         "targets_cat_num_dist": [("day", "total_bill"), ("time", "tip"), ("smoker", "total_bill")],
         "targets_2D_num": [("total_bill", "tip"), ("size", "total_bill")],
         "targets_3D_num": [("total_bill", "tip", "size")],
         "targets_multi_num": [["total_bill", "tip", "size"]],
         "targets_cat_num_comp": [("day", "total_bill"), ("time", "tip"), ("smoker", "tip")],
         "targets_2cat_num": [("day", "time", "total_bill"), ("day", "sex", "tip"), ("time", "smoker", "tip"), ("sex", "smoker", "total_bill")],
         "targets_radar": [("day", ["total_bill", "tip", "size"]), ("time", ["total_bill", "tip", "size"]), ("sex", ["total_bill", "tip", "size"]), ("smoker", ["total_bill", "tip", "size"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},
         
        {"dataset_name": "crashes", "domain": "Insurance", "data": df_crashes,
         "targets_1D_num": ["ins_premium", "speeding", "alcohol"], 
         "targets_cat_num_dist": [],
         "targets_2D_num": [("speeding", "alcohol"), ("alcohol", "ins_premium"), ("speeding", "ins_premium")], 
         "targets_3D_num": [("speeding", "alcohol", "ins_premium"), ("not_distracted", "no_previous", "total"), ("alcohol", "ins_losses", "ins_premium")],
         "targets_multi_num": [["total", "speeding", "alcohol", "not_distracted"], ["no_previous", "ins_premium", "ins_losses"], ["speeding", "alcohol", "ins_premium", "ins_losses"], ["total", "speeding", "alcohol", "not_distracted", "no_previous", "ins_premium", "ins_losses"]],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [],
         "targets_radar": [("abbrev", ["speeding", "alcohol", "not_distracted", "no_previous"]), ("abbrev", ["speeding", "alcohol", "ins_premium"]), ("abbrev", ["no_previous", "ins_losses", "total"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},

        # AUTOMOTIVE AND ASTRONOMY (mpg, planets) -> Used frequently for Time Series

        {"dataset_name": "mpg", "domain": "Automotive", "data": df_mpg,
         "targets_1D_num": ["mpg", "horsepower", "weight"], 
         "targets_cat_num_dist": [("origin", "mpg"), ("origin", "horsepower"), ("cylinders", "weight")],
         "targets_2D_num": [("horsepower", "mpg"), ("weight", "mpg"), ("displacement", "mpg")], 
         "targets_3D_num": [("horsepower", "weight", "mpg"), ("displacement", "weight", "mpg"), ("acceleration", "horsepower", "mpg")],
         "targets_multi_num": [["mpg", "cylinders", "displacement", "horsepower"], ["weight", "acceleration", "model_year"], ["mpg", "displacement", "horsepower", "weight", "acceleration"], ["cylinders", "displacement", "horsepower", "weight"]],
         "targets_cat_num_comp": [("origin", "mpg"), ("origin", "horsepower"), ("cylinders", "mpg")],
         "targets_2cat_num": [("origin", "cylinders", "mpg")],
         "targets_radar": [("origin", ["mpg", "horsepower", "weight", "acceleration"]), ("cylinders", ["mpg", "horsepower", "weight", "acceleration"]), ("model_year", ["mpg", "horsepower", "weight", "acceleration"])],
         # Evolution of cars during the years
         "targets_time_num": [("model_year", "mpg"), ("model_year", "horsepower"), ("model_year", "weight"), ("model_year", "acceleration")], 
         "targets_time_num_cat": [("model_year", "mpg", "origin"), ("model_year", "horsepower", "origin"), ("model_year", "weight", "origin"), ("model_year", "acceleration", "origin"), ("model_year", "mpg", "cylinders"), ("model_year", "horsepower", "cylinders")], 
         "targets_candlestick": []},

        {"dataset_name": "planets", "domain": "Astronomy", "data": df_planets,
         "targets_1D_num": ["distance", "mass"], 
         "targets_cat_num_dist": [],
         "targets_2D_num": [("mass", "distance"), ("orbital_period", "distance")], 
         "targets_3D_num": [("mass", "distance", "orbital_period"), ("mass", "orbital_period", "distance")],
         "targets_multi_num": [["number", "orbital_period", "mass", "distance"], ["mass", "distance", "year"]],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [],
         "targets_radar": [("method", ["mass", "distance", "orbital_period"])],
         # Discoveries over the years
         "targets_time_num": [("year", "distance"), ("year", "mass"), ("year", "orbital_period")], 
         "targets_time_num_cat": [("year", "distance", "method"), ("year", "mass", "method"), ("year", "orbital_period", "method")], 
         "targets_candlestick": []},

        # PURE TIME SERIES (taxis, healthexp, flights, fmri, dots, attention, dowjones)

        {"dataset_name": "taxis", "domain": "Mobility", "data": df_taxis,
         "targets_1D_num": ["fare", "distance"], 
         "targets_cat_num_dist": [("payment", "fare"), ("pickup_borough", "distance"), ("color", "tip")],
         "targets_2D_num": [("distance", "fare"), ("fare", "tip"), ("distance", "tip")], 
         "targets_3D_num": [("distance", "fare", "tip"), ("fare", "tolls", "total"), ("distance", "tolls", "fare"), ("tolls", "tip", "total"), ("distance", "total", "fare")],
         "targets_multi_num": [["passengers", "distance", "fare"], ["tip", "tolls", "total"], ["distance", "fare", "tip", "tolls", "total"], ["passengers", "distance", "fare", "tip", "tolls", "total"]],
         "targets_cat_num_comp": [("payment", "fare"), ("pickup_borough", "total"), ("color", "distance")], 
         "targets_2cat_num": [("pickup_borough", "payment", "fare"), ("pickup_borough", "color", "distance"), ("dropoff_borough", "payment", "tip"), ("color", "payment", "total")],
         "targets_radar": [("pickup_borough", ["distance", "fare", "tip", "total"]), ("payment", ["distance", "fare", "tip", "total"]), ("color", ["distance", "fare", "tip", "total"]), ("dropoff_borough", ["distance", "fare", "tip", "total"])],
         "targets_time_num": [], 
         "targets_time_num_cat": [], 
         "targets_candlestick": []},

        {"dataset_name": "healthexp", "domain": "Healthcare", "data": df_healthexp,
         "targets_1D_num": ["Spending_USD"], 
         "targets_cat_num_dist": [], 
         "targets_2D_num": [], 
         "targets_3D_num": [("Year", "Spending_USD", "Life_Expectancy")],
         "targets_multi_num": [["Year", "Spending_USD", "Life_Expectancy"]],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [],
         "targets_radar": [],
         "targets_time_num": [("Year", "Spending_USD"), ("Year", "Life_Expectancy")],
         "targets_time_num_cat": [("Year", "Spending_USD", "Country"), ("Year", "Life_Expectancy", "Country")],
         "targets_candlestick": []},

        {"dataset_name": "flights", "domain": "Aviation", "data": df_flights,
         "targets_1D_num": ["passengers"], 
         "targets_cat_num_dist": [], 
         "targets_2D_num": [], 
         "targets_3D_num": [], 
         "targets_multi_num": [],
         "targets_cat_num_comp": [("month", "passengers")], 
         "targets_2cat_num": [], 
         "targets_radar": [],
         "targets_time_num": [("year", "passengers"), ("month", "passengers")], 
         "targets_time_num_cat": [("year", "passengers", "month"), ("month", "passengers", "year")],
         "targets_candlestick": []},

        {"dataset_name": "fmri", "domain": "Biology", "data": df_fmri,
         "targets_1D_num": [], 
         "targets_cat_num_dist": [("event", "signal"), ("region", "signal")], 
         "targets_2D_num": [], 
         "targets_3D_num": [], 
         "targets_multi_num": [["timepoint", "signal"]],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [("region", "event", "signal"), ("event", "region", "signal")], 
         "targets_radar": [],
         "targets_time_num": [("timepoint", "signal")], 
         "targets_time_num_cat": [("timepoint", "signal", "region"), ("timepoint", "signal", "event"), ("timepoint", "signal", "subject")],
         "targets_candlestick": []},

        {"dataset_name": "dots", "domain": "Biology", "data": df_dots,
         "targets_1D_num": [], 
         "targets_cat_num_dist": [], 
         "targets_2D_num": [], 
         "targets_3D_num": [], 
         "targets_multi_num": [],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [("choice", "align", "firing_rate"), ("align", "choice", "firing_rate")], 
         "targets_radar": [],
         "targets_time_num": [("time", "firing_rate"), ("time", "coherence")], 
         "targets_time_num_cat": [("time", "firing_rate", "choice"), ("time", "firing_rate", "align"), ("time", "firing_rate", "coherence"), ("time", "coherence", "choice"), ("time", "coherence", "align")],
         "targets_candlestick": []},

        {"dataset_name": "attention", "domain": "Psychology", "data": df_attention,
         "targets_1D_num": [], 
         "targets_cat_num_dist": [("attention", "score")], 
         "targets_2D_num": [], 
         "targets_3D_num": [], 
         "targets_multi_num": [],
         "targets_cat_num_comp": [("attention", "score")], 
         "targets_2cat_num": [], 
         "targets_radar": [],
         "targets_time_num": [("solutions", "score")], 
         "targets_time_num_cat": [("solutions", "score", "attention"), ("solutions", "score", "subject")],
         "targets_candlestick": []},
         
        {"dataset_name": "dowjones", "domain": "Finance", "data": df_dowjones,
         "targets_1D_num": [], 
         "targets_cat_num_dist": [], 
         "targets_2D_num": [], 
         "targets_3D_num": [], 
         "targets_multi_num": [],
         "targets_cat_num_comp": [], 
         "targets_2cat_num": [], 
         "targets_radar": [],
         "targets_time_num": [("Date", "Price")], 
         "targets_time_num_cat": [],
         "targets_candlestick": []}

    ]

    # Adding the finantial datasets for time series

    if "AAPL" in df_fin_time:
        generation_config.append(
            {"dataset_name": "aapl_time", "domain": "Finance", "data": df_fin_time["AAPL"], 
             "targets_1D_num": [], 
             "targets_cat_num_dist": [], 
             "targets_2D_num": [], 
             "targets_3D_num": [], 
             "targets_multi_num": [], 
             "targets_cat_num_comp": [], 
             "targets_2cat_num": [], 
             "targets_radar": [], 
             "targets_time_num_cat": [], 
             "targets_candlestick": [], 
             "targets_time_num": [("Date", "Close"), ("Date", "Volume")]}
            )
    if "GOOGL" in df_fin_time:
        generation_config.append(
            {"dataset_name": "googl_time", "domain": "Finance", "data": df_fin_time["GOOGL"], 
             "targets_1D_num": [], 
             "targets_cat_num_dist": [], 
             "targets_2D_num": [], 
             "targets_3D_num": [], 
             "targets_multi_num": [], 
             "targets_cat_num_comp": [], 
             "targets_2cat_num": [], 
             "targets_radar": [], 
             "targets_time_num_cat": [], 
             "targets_candlestick": [], 
             "targets_time_num": [("Date", "Close"), ("Date", "Volume")]}
            )
    if "MSFT" in df_fin_time:
        generation_config.append(
            {"dataset_name": "msft_time", "domain": "Finance", "data": df_fin_time["MSFT"], 
             "targets_1D_num": [], 
             "targets_cat_num_dist": [], 
             "targets_2D_num": [], 
             "targets_3D_num": [], 
             "targets_multi_num": [], 
             "targets_cat_num_comp": [], 
             "targets_2cat_num": [], 
             "targets_radar": [], 
             "targets_time_num_cat": [], 
             "targets_candlestick": [], 
             "targets_time_num": [("Date", "Close"), ("Date", "Volume")]}
            )
    if "AMZN" in df_fin_time:
        generation_config.append(
            {"dataset_name": "amzn_time", "domain": "Finance", "data": df_fin_time["AMZN"], 
             "targets_1D_num": [], 
             "targets_cat_num_dist": [], 
             "targets_2D_num": [], 
             "targets_3D_num": [], 
             "targets_multi_num": [], 
             "targets_cat_num_comp": [], 
             "targets_2cat_num": [], 
             "targets_radar": [], 
             "targets_time_num_cat": [], 
             "targets_candlestick": [], 
             "targets_time_num": [("Date", "Close")]}
            )

    ##################### ADDING THE FINANCIAL DATASETS FOR CANDLESTICKS #####################

    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", # Technology and Motor
        "META", "NVDA", "NFLX", "INTC", "CSCO",  # Techonolgy and Network
        "JPM", "V", "GS",                        # Finance
        "JNJ", "PFE",                            # Pharmaceuticals
        "WMT", "KO", "NKE",                      # Consumption
        "BA", "XOM"                              # Industry and Energy
    ]

    for ticker in tickers:
        try:
            df_stock = yf.download(ticker, start="2025-09-01", end="2026-01-01", progress=False).reset_index()
            if isinstance(df_stock.columns, pd.MultiIndex):
                df_stock.columns = df_stock.columns.droplevel(1) 
            df_stock.rename(columns={'index': 'Date', 'Date': 'Date'}, inplace=True)
            generation_config.append({
                "dataset_name": f"{ticker.lower()}_stock", "domain": "Finance", "data": df_stock,
                "targets_1D_num": [], 
                "targets_cat_num_dist": [], 
                "targets_2D_num": [], 
                "targets_3D_num": [], 
                "targets_multi_num": [], 
                "targets_cat_num_comp": [], 
                "targets_2cat_num": [], 
                "targets_radar": [], 
                "targets_time_num": [], 
                "targets_time_num_cat": [],
                "targets_candlestick": [("Date", "Open", "High", "Low", "Close")]
            })
        except Exception as e:
            print(f"ERROR: Could not download the data for {ticker}: {e}")

    ##################### CASE GENERATION LOOP EXECUTION #####################
    
    all_cases = []
    current_id = 1  # We initialize the ids at 000001
    
    for config in generation_config:

        df = config.get("data")
        domain = config.get("domain")
        
        # 1. DISTRIBUTION (1D Num)

        for num_col in config.get("targets_1D_num", []):
            all_cases.append(generate_histogram(df, num_col, domain, current_id)); current_id += 1
            all_cases.append(generate_kde(df, num_col, domain, current_id)); current_id += 1
            
        # 1. DISTRIBUTION (Cat + Num)

        for cat, num in config.get("targets_cat_num_dist", []):
            all_cases.append(generate_boxplot(df, cat, num, domain, current_id)); current_id += 1
            all_cases.append(generate_violin(df, cat, num, domain, current_id)); current_id += 1
            all_cases.append(generate_swarm(df, cat, num, domain, current_id)); current_id += 1

        # 2. RELATION AND CORRELATION

        for x, y in config.get("targets_2D_num", []):
            all_cases.append(generate_scatter(df, x, y, domain, current_id)); current_id += 1
            all_cases.append(generate_scatter_reg(df, x, y, domain, current_id)); current_id += 1
            
        for x, y, size in config.get("targets_3D_num", []):
            all_cases.append(generate_bubble(df, x, y, size, domain, current_id)); current_id += 1
            
        for cols in config.get("targets_multi_num", []):
            all_cases.append(generate_heatmap(df, cols, domain, current_id)); current_id += 1

        # 3. COMPARISON AND COMPOSITION

        for cat, num in config.get("targets_cat_num_comp", []):
            all_cases.append(generate_barplot(df, cat, num, domain, current_id)); current_id += 1
            all_cases.append(generate_pie(df, cat, num, domain, current_id)); current_id += 1
            all_cases.append(generate_donut(df, cat, num, domain, current_id)); current_id += 1
            
        for cat1, cat2, num in config.get("targets_2cat_num", []):
            all_cases.append(generate_grouped_bar(df, cat1, cat2, num, domain, current_id)); current_id += 1
            all_cases.append(generate_stacked_bar(df, cat1, cat2, num, domain, current_id)); current_id += 1
            all_cases.append(generate_treemap(df, cat1, cat2, num, domain, current_id)); current_id += 1

        for cat_entity, num_cols in config.get("targets_radar", []):
            all_cases.append(generate_radar(df, cat_entity, num_cols, domain, current_id)); current_id += 1

        # 4. TIME SERIES

        for time_col, num in config.get("targets_time_num", []):
            all_cases.append(generate_line(df, time_col, num, domain, current_id)); current_id += 1
            all_cases.append(generate_area(df, time_col, num, domain, current_id)); current_id += 1
            all_cases.append(generate_line_ci(df, time_col, num, domain, current_id)); current_id += 1
            
        for time_col, num, cat in config.get("targets_time_num_cat", []):
            all_cases.append(generate_multiple_line(df, time_col, num, cat, domain, current_id)); current_id += 1
            all_cases.append(generate_stacked_area(df, time_col, num, cat, domain, current_id)); current_id += 1

        for time_col, open_c, high_c, low_c, close_c in config.get("targets_candlestick", []):
            all_cases.append(generate_candlestick(df, time_col, open_c, high_c, low_c, close_c, domain, current_id)); current_id += 1

    # We save the final CSV with all the generated cases
    if all_cases:
        df_final = pd.DataFrame(all_cases)
        df_final.to_csv("case_base_prev.csv", index=False)
        print(f"Generation completed: {len(df_final)} graphs generated and saved.")
    else:
        print("ERROR: No graphs were generated.")

if __name__ == "__main__":
    main()