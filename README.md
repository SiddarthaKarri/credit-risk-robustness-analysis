# Beyond Accuracy: Credit Risk Model Robustness under Distribution Shift

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Research%20Complete-success)
![License](https://img.shields.io/badge/License-MIT-green)

This repository hosts the code and analysis for a research study evaluating how credit risk prediction models degrade under temporal distribution shift. We move beyond standard accuracy metrics to quantify **robustness**, **financial impact**, and **feature drift** using the Lending Club dataset.

## 📊 Key Research Contributions
- **Drift Quantification**: Formally quantified covariate shift using **Population Stability Index (PSI)** and Wasserstein Distance. Identified `revol_util` as a primary driver of drift (PSI > 0.25).
- **Robustness Ratios**: Proposed a metric to measure performance retention ($AUC_{OOD} / AUC_{ID}$) across economic cycles.
- **Cost-Sensitive Learning**: Evaluated models using a financial cost function (False Negatives cost 10x False Positives), revealing that AUC-optimal models are not always profit-optimal.
- **Statistical Rigor**: Verified results with **Bootstrapped Confidence Intervals (95% CI)**.

## 📂 Repository Structure

```
├── data/               # Raw and processed datasets (Gitignored)
├── notebooks/          # Interactive research notebooks
│   ├── 01_eda.ipynb             # Exploratory Analysis & Drift Detection
│   ├── 02_preprocessing.ipynb   # Feature Engineering & Imputation
│   ├── 03_models.ipynb          # Model Training (LR, XGB, RF, LGBM)
│   ├── 04_evaluation.ipynb      # Robustness & Cost Analysis
│   └── 05_drift_analysis.ipynb  # PSI & Wasserstein Metric Calculation
├── paper/              # LaTeX source for the conference paper
├── results/            # Generated artifacts
│   ├── figures/        # HD plots (Drift KDEs, Cost Curves)
│   ├── models/         # Serialized model binaries
│   └── tables/         # CSV results for paper integration
├── src/                # Utility scripts for metrics and data loading
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Google Colab

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/SiddarthaKarri/credit-risk-robustness-analysis.git
    cd credit-risk-robustness-analysis
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiment
The research workflow is divided into 5 sequential notebooks:

1.  **Exploratory Data Analysis (`01_eda.ipynb`)**: Downloads Lending Club data and visualizes initial distributions.
2.  **Preprocessing (`02_preprocessing.ipynb`)**: Cleans data, handles missing values, and creates In-Time (Train) vs Out-of-Time (OOD) splits.
3.  **Model Training (`03_models.ipynb`)**: Trains XGBoost, LightGBM, Random Forest, and Logistic Regression baselines.
4.  **Evaluation (`04_evaluation.ipynb`)**: Generates robustness ratios, cost curves, and SHAP stability plots.
5.  **Drift Analysis (`05_drift_analysis.ipynb`)**: Calculates PSI for all features to diagnose the root cause of degradation.

## 📈 Key Results

| Model | ID AUC (2014-16) | OOD AUC (2018-19) | Robustness Ratio | Cost (10:1 Ratio) |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.724 | 0.699 | **0.965** | 41,098 |
| **XGBoost** | 0.727 | 0.698 | 0.960 | **40,990** |
| **Logistic Regression** | 0.716 | 0.686 | 0.958 | 42,069 |

*While LightGBM is the most robust, XGBoost provides the lowest financial loss in a high-risk lending scenario.*

## View Paper
```markdown
[**Click here to view Research Paper**](paper/paper.pdf)
```

## 📜 Citation  
If you use this code or analysis, please refer to the paper in the `paper/` directory.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
