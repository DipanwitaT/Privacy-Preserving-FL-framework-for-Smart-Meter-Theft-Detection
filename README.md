# Privacy-Preserving-FL-framework-for-Smart-Meter-Theft-Detection
Smart Grid Application

This repository implements a complete Federated Learning (FL) framework for smart-meter electricity theft detection using the SGCC (State Grid Corporation of China) consumption dataset.  
The system evaluates the trade-offs among:
- Utility performance (Accuracy, Precision, Recall, F1-score)  
- Privacy guarantees (DP, RDP, zCDP, MA)  
- Communication cost  
- Federated optimization algorithms (FedAvg, FedProx, SCAFFOLD)  
- Different client data distributions (IID and non-IID Dirichlet α = 0.5)

The project includes a full grid-search engine for benchmarking all combinations of  algorithm, privacy accountant, and noise multiplier 
and produces: (1) Master CSV of all results  (2) Per-algorithm and per-accountant result files and (3) Automated trade-off plots for visualization  

The SGCC electricity consumption dataset ((State Grid Corporation of China)) is publicly available and can be downloaded from https://www.kaggle.com/datasets/bensalem14/sgcc-dataset
After downloading:

1. Extract the CSV file(s).
2. Place the primary CSV (e.g., `data set.csv`) in the project root directory  
   or specify your own path using the `SGCC_DATA_PATH` environment variable.

Example:
```bash
export SGCC_DATA_PATH="/path/to/data set.csv"



The Repository Structure is as follows:-
├── data_loader.py          # Dataset loading, preprocessing, and client partitioning
├── model.py                # Lightweight MLP model for edge-device suitability
├── fl_core.py              # FedAvg, FedProx, SCAFFOLD + DP aggregation
├── dp_utils.py             # DP, RDP, zCDP, MA privacy accountants
├── plots.py                # Privacy–utility–communication trade-off visualizations
├── main.py                 # Single-run evaluation over multiple σ values
├── grid_search.py          # Full sweep over algorithms × accountants × σ
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

```

# 1. Clone Repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2.Install python dependencies
pip install -r requirements.txt

Include:
numpy
pandas
matplotlib
torch
scikit-learn
opacus

# Running a Standard Experiment
Example (single sweep in main.py)
python main.py \
    --partition iid \
    --algo fedavg \
    --dp rdp \
    --sigmas 0.0,0.5,1.0,2.0

# Parameters

--partition - iid or non-iid
--alpha - Dirichlet α for non-IID (default 0.5)
--algo - fedavg, fedprox, scaffold
--dp - dp, rdp, zcdp, ma
--sigmas - noise multipliers for DP

# Running the Full Grid Search

The grid search evaluates all combinations of:

FL algorithm
Privacy accountant
Noise multiplier σ

# Run:
python grid_search.py \
    --partition non-iid \
    --alpha 0.5 \
    --algos fedavg,fedprox,scaffold \
    --dps dp,rdp,zcdp,ma \
    --sigmas 0.0,0.5,1.0,2.0

# Output:

master_results_partition-*.csv
Per-algorithm accountant CSVs
Automatic trade-off plots:
Privacy vs Utility
Privacy vs Communication
Utility vs Communication
Privacy vs F1
Best configuration bar-plots

## All results are stored in
D:/Dipanwita/privacy_smartgrid/opacus_results/
This is our custom OUT_DIR

#Visualizations
The framework generates multiple figures such as:
Privacy–Utility (ε vs Accuracy)
Privacy–Communication (ε vs MB)
Utility–Communication (Accuracy vs MB)
Privacy–F1
Best-test-metrics bar plot
These plots help identify Pareto-optimal trade-offs for deployment.

# Why a Lightweight MLP Model?

Smart meters and edge devices have:
low memory (128–512 kB),
limited CPU power,
strict energy constraints.

Our MLP has 
1. small parameter count,
2. enables real-time on-device inference,
3. minimizes communication overhead in FL,
4. is robust under DP noise
5. is compatible with non-IID client data
6. ideal for large-scale smart-grid deployments



