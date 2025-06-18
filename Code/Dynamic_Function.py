import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_dynamic_year_shipment(df_original, year, scaler):
    """
    Prepares a future shipment dataset for any year by interpolating if needed
    and applying correct feature normalization using previously fit scaler.

    Returns:
        - df_future: DataFrame ready for inference
        - true_values: Ground-truth cost if available, else None
        - original_distance: unscaled distances (for evaluation/agent inspection)
    """
    # Step 1: Extract available year columns
    available_years = sorted([int(col.split('_')[1]) for col in df_original.columns if 'tons_' in col])

    # Step 2: Determine which columns to use or interpolate
    if year in available_years:
        tons_col = f"tons_{year}"
        tmiles_col = f"tmiles_{year}"
        value_col = f"value_{year}" if f"value_{year}" in df_original.columns else None
        df_future = df_original.copy()
    else:
        past_years = [y for y in available_years if y < year]
        future_years = [y for y in available_years if y > year]

        if not past_years or not future_years:
            nearest_year = min(available_years, key=lambda y: abs(y - year))
            print(f"⚠️ Year {year} out of range. Using closest available year {nearest_year}.")
            tons_col = f"tons_{nearest_year}"
            tmiles_col = f"tmiles_{nearest_year}"
            value_col = f"value_{nearest_year}" if f"value_{nearest_year}" in df_original.columns else None
            df_future = df_original.copy()
        else:
            y0, y1 = max(past_years), min(future_years)
            w = (year - y0) / (y1 - y0)

            df_future = df_original.copy()
            df_future[f"tons_{year}"] = (1 - w) * df_future[f"tons_{y0}"] + w * df_future[f"tons_{y1}"]
            df_future[f"tmiles_{year}"] = (1 - w) * df_future[f"tmiles_{y0}"] + w * df_future[f"tmiles_{y1}"]
            value_col = f"value_{year}" if f"value_{year}" in df_original.columns else None

            tons_col = f"tons_{year}"
            tmiles_col = f"tmiles_{year}"

    # Save original (unscaled) distance
    original_distance = df_future[tmiles_col].copy()

    # Step 3: Drop all other tons/tmiles/value columns
    drop_cols = [c for c in df_future.columns if (
        c.startswith("tons_") or c.startswith("tmiles_") or c.startswith("value_")
    ) and c not in [tons_col, tmiles_col, value_col]]

    df_future = df_future.drop(columns=drop_cols)

    # Step 4: Rename to match scaler expect
	
	# Step 1: Prepare interpolated dataset for future year (e.g., 2032)
df_2032, true_costs, _ = prepare_dynamic_year_shipment(df_original, year=2050, scaler=scaler_features)

# Step 2: Define mode-specific distances (optional but powerful)
custom_distances = {
    "dms_mode_Truck": 1200.0,
    "dms_mode_Air": 8000.0,
    "dms_mode_Rail": 1000.0,
    "dms_mode_Water": 1400.0,
    "dms_mode_Pipeline": 1100.0,
    "dms_mode_Multiple_Modes_and_Mail": 950.0,
    "dms_mode_Other_and_Unknown": 1050.0
}

# Step 3: Initialize the goal-driven agent
agent = GoalDrivenAgent(
    model=model,
    device=device,
    feature_columns=X_train.columns,
    cost_scaler=scaler_value,
    feature_scaler=scaler_features,
    year=2050,
    goal="fastest",           # Can be "cheapest", "fastest", or "balanced"
    alpha=0.7,
    beta=0.3,
    custom_distances=custom_distances
)

# Step 4: Test the agent on a sample shipment from 2032
sample_2032 = df_2032.sample(n=1, random_state=42)
agent.decide(sample_2032)
ations (trained on 2012 columns)
    df_scaled = df_future.rename(columns={tons_col: "tons_2012", tmiles_col: "tmiles_2012"})

    # Step 5: Scale
    df_scaled[["tons_2012", "tmiles_2012"]] = scaler.transform(df_scaled[["tons_2012", "tmiles_2012"]])

    # ⚠️ Do NOT rename them back – leave as tons_2012, tmiles_2012 for agent to work correctly

    # Step 6: Get true values if present
    true_values = df_original[value_col] if value_col else None

    return df_scaled, true_values, original_distance

	
	# Step 1: Prepare interpolated dataset for future year (e.g., 2032)
df_2032, true_costs, _ = prepare_dynamic_year_shipment(df_original, year=2050, scaler=scaler_features)

# Step 2: Define mode-specific distances (optional but powerful)
custom_distances = {
    "dms_mode_Truck": 1200.0,
    "dms_mode_Air": 8000.0,
    "dms_mode_Rail": 1000.0,
    "dms_mode_Water": 1400.0,
    "dms_mode_Pipeline": 1100.0,
    "dms_mode_Multiple_Modes_and_Mail": 950.0,
    "dms_mode_Other_and_Unknown": 1050.0
}

# Step 3: Initialize the goal-driven agent
agent = GoalDrivenAgent(
    model=model,
    device=device,
    feature_columns=X_train.columns,
    cost_scaler=scaler_value,
    feature_scaler=scaler_features,
    year=2050,
    goal="fastest",           # Can be "cheapest", "fastest", or "balanced"
    alpha=0.7,
    beta=0.3,
    custom_distances=custom_distances
)

# Step 4: Test the agent on a sample shipment from 2032
sample_2032 = df_2032.sample(n=1, random_state=42)
agent.decide(sample_2032)
