# Re-import necessary libraries after state reset
import torch
import pandas as pd
from typing import Optional, Dict
from sklearn.preprocessing import MinMaxScaler

# Re-define the updated GoalDrivenAgent class after kernel reset
class GoalDrivenAgent:
    def __init__(
        self,
        model,
        device,
        feature_columns,
        cost_scaler,
        feature_scaler,
        year: int,
        goal: str = "cheapest",
        alpha: float = 0.7,
        beta: float = 0.3,
        custom_distances: Optional[Dict[str, float]] = None  # key: mode column name
    ):
        self.model = model
        self.device = device
        self.feature_columns = feature_columns
        self.cost_scaler = cost_scaler
        self.feature_scaler = feature_scaler
        self.year = year
        self.goal = goal
        self.alpha = alpha
        self.beta = beta
        self.custom_distances = custom_distances or {}

    def decide(self, shipment: pd.DataFrame):
        shipment = shipment.copy()
        base_features = shipment.drop(columns=[col for col in shipment if col.startswith("dms_mode_")])
        mode_columns = [col for col in self.feature_columns if col.startswith("dms_mode_")]

        decisions = []

        for mode in mode_columns:
            sample = pd.DataFrame(columns=self.feature_columns)
            sample.loc[0] = 0
            sample.update(base_features)
            sample[mode] = 1

            # Predict cost
            tensor_sample = torch.tensor(sample.values, dtype=torch.float32).to(self.device)
            normalized_cost = self.model(tensor_sample).cpu().detach().numpy()
            cost = self.cost_scaler.inverse_transform(normalized_cost.reshape(1, -1))[0][0]

            # Determine distance
            if mode in self.custom_distances:
                distance = self.custom_distances[mode]  # Use custom override
            else:
                tmiles_col = f"tmiles_{self.year}"
                tons_col = f"tons_{self.year}"
                distance_scaled = shipment[tmiles_col].values[0]
                tons_scaled = shipment[tons_col].values[0]
                unscaled_values = self.feature_scaler.inverse_transform([[tons_scaled, distance_scaled]])
                distance = unscaled_values[0][1]  # Extract unscaled distance

            # Compute reward
            if self.goal == "cheapest":
                reward = -cost
            elif self.goal == "fastest":
                reward = -distance
            elif self.goal == "balanced":
                reward = -self.alpha * cost - self.beta * distance
            else:
                raise ValueError("Invalid goal specified. Use 'cheapest', 'fastest', or 'balanced'.")

            decisions.append((mode, cost, distance, reward))

        best_mode = max(decisions, key=lambda x: x[3])

        print("\n🧠 Goal-Driven Agent Decision Summary:")
        for mode, cost, distance, reward in decisions:
            print(f"{mode}: Cost=${cost:.2f}, Distance={distance:.2f}, Reward={reward:.2f}")
        print(f"\n✅ Selected Mode: {best_mode[0]} based on goal '{self.goal}'")

        return best_mode[0]

custom_distances = {
    "dms_mode_Truck": 12000.5,
    "dms_mode_Air": 850.0,
    "dms_mode_Rail": 1000.0,
    "dms_mode_Water": 1450.2
}


agent = GoalDrivenAgent(
    model=model,
    device=device,
    feature_columns=X_train.columns,
    cost_scaler=scaler_value,
    feature_scaler=scaler_features,
    year=3000,
    goal="balanced",  # or "cheapest", "fastest"
    alpha=0.6,
    beta=0.4,
    custom_distances=custom_distances  # Optional
)
sample_2032 = df_original.sample(n=1, random_state=42)
agent.decide(sample_2032)