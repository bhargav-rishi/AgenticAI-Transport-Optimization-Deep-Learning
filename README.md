# Agentic AI for Transport Mode Optimization

![Built With](https://img.shields.io/badge/Built%20With-Python%20%7C%20Agentic%20AI%20%7C%20PyTorch%20%7C%20Deep%20Learning%20%7C%20Deep%20Reinforcement%20Learning-blue)
![Language](https://img.shields.io/badge/Language-Python-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Project Overview

This project integrates **deep learning** and **agentic decision-making** to recommend the most optimal freight transport mode (e.g., Truck, Rail, Air) for a given shipment. Built on the **Freight Analysis Framework (FAF)** dataset, it supports both historical and **future predictions** (e.g., year 2050) using a combination of:

- **Neural Network** for shipment cost prediction  
- **Goal-Driven Agent** to choose "cheapest", "fastest", or "balanced" modes  
- **Q-Learning Agent** that learns from simulated rewards  
- **Dynamic Forecasting** to interpolate data for unseen years
  
The project project was developed as part of the Deep Learning course at the **University of South Florida – Muma College of Business**.

---

## Dataset (FAF4)
Sourced from the U.S. Department of Transportation - https://www.bts.gov/faf/faf4
Includes:
- `dms_orig`, `dms_dest`: Origin/Destination zones
- `tons_2012`, `value_2012`, `tmiles_2012`: Shipment weight, cost, and distance
- `dms_mode`: Transport mode (one-hot encoded)
- `trade_type`: Filtered for domestic shipments only

---

## Technologies & Methods

| Component | Description |
|----------|-------------|
| **Neural Network** | PyTorch MLP model predicting shipment cost |
| **Scaler** | MinMaxScaler for normalizing inputs (tons, miles) |
| **Agent** | Goal-driven logic chooses optimal mode per shipment |
| **Q-Learning** | RL agent learns optimal policy using synthetic rewards |
| **Interpolation** | Supports feature projection for years like 2032 or 2050 |

---

## File Structure
<pre>
Code/
├── AgenticAI_Transport_Optimization_Final.ipynb # Main notebook
├── NN_Model.py # Deep learning pipeline (cost prediction)
├── Goal_Driven_Agent.py # Goal-oriented agent logic
├── Dynamic_Function.py # Year-based forecasting for 2050, etc.
├── Q_Learning.py # Reinforcement learning implementation
├── presentation/
│ └── Presentation.docx # Project presentation file
|README.md
</pre>
---

## Key Results

| Model          | Description                              |
|----------------|------------------------------------------|
| Neural Network | Achieved stable convergence (MSE-based) |
| Goal-Driven Agent | Accurately selects best mode per goal |
| Q-Learning     | Learns optimal policies over 5,000 episodes |

---

## Use Case Example

> Predict the best transport mode for a 2050 shipment from Zone 101 to Zone 205 based on a "balanced" goal.  
> Agent interpolates projected features, evaluates all modes, and selects based on combined cost-distance reward.

---

## Author

**Bhargav Rishi Medisetti**  
Muma College of Business – University of South Florida




