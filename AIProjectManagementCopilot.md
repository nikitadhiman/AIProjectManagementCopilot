# AI-Driven Delay Prediction and Developer Recommendation System

_(AI Project Management Copilot)_

This project implements an **AI Project Management Copilot** that helps software teams:

- Predict which tasks are at **risk of delay**
- Understand **why** they are risky (explanations)
- Get **developer recommendations** for each task based on expertise

The system combines **classical ML**, **explainable AI**, and **graph neural networks** into a single, interactive **Streamlit dashboard** that can be run in **Google Colab** with **ngrok**.

---

## Key Features

- **Synthetic Scrum-like dataset generator**
  - Tasks: story points, components, blockers, reopens, etc.
  - Developers: skills, experience, historical tasks
- **Delay Risk Prediction** using **XGBoost**
  - Outputs probability of delay for each task
- **Explainable AI with SHAP**
  - Feature-level explanations for each prediction
  - Highlights what increased / decreased delay risk
- **Developer Recommendation with GraphSAGE GNN**
  - Bipartite graph of developers ↔ tasks
  - Learns embeddings and suggests best-fit developers
- **Streamlit Dashboard**
  - View tasks, risk levels, explanations, and top-K recommended developers
  - Designed to simulate an **engineering manager’s decision assistant**
- **Colab-friendly setup**
  - Single notebook
  - CPU-only GNN (no `pyg-lib` / `torch-sparse` required)
  - Optional public URL via **ngrok**

---

## 🧱 Tech Stack

- **Language**: Python 3
- **ML / DL**:
  - [XGBoost](https://github.com/dmlc/xgboost)
  - [PyTorch](https://pytorch.org/)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (CPU only)
- **Explainability**: [SHAP](https://github.com/shap/shap)
- **Web UI**: [Streamlit](https://streamlit.io/)
- **Tunneling (for Colab)**: [ngrok](https://ngrok.com/)

---

## 📁 Project Structure (Conceptual)

If you export the notebook into scripts, your repo might look like:

```text
.
├── ai_pm_copilot_colab.ipynb     # Main notebook (all code in one place)
├── app.py                        # (Optional) Streamlit app extracted from the notebook
├── data/
│   ├── synthetic_tasks.csv       # Generated synthetic task dataset
│   └── synthetic_developers.csv  # Generated synthetic developer dataset
├── models/
│   ├── xgb_delay_model.json      # Saved XGBoost model (optional)
│   └── gnn_embeddings.pt         # Saved GNN embeddings (optional)
└── README.md                     # This file
```
