import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_data, split_data_by_time

# Configuration
DATA_PATH = r'c:\Users\sidda\OneDrive\Desktop\sol\credit-risk-robustness\data\raw\accepted_2007_to_2018Q4.csv'
FIGURES_DIR = r'c:\Users\sidda\OneDrive\Desktop\sol\credit-risk-robustness\results\figures'

def perform_eda():
    print("Loading data...")
    # Load only necessary columns for initial EDA to save memory, or load all if machine permits. 
    # For now, let's load a subset of likely important columns + dates/target
    cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'issue_d', 'loan_status', 'purpose', 'dti', 'fico_range_low', 'fico_range_high',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
        'pub_rec_bankruptcies'
    ]
    
    # USE SAMPLING FOR LOCAL DEV (50k rows) to avoid lag
    # We can remove this limit for the final run or on Colab
    try:
        df = load_data(DATA_PATH, cols=cols, nrows=50000)
    except FileNotFoundError:
        print(f"Data not found at {DATA_PATH}. Please ensure download is complete.")
        return

    print(f"Data Loaded: {df.shape}")
    
    # Filter target
    # We only care about 'Fully Paid' and 'Charged Off' for binary classification usually
    # Or 'Default'
    target_mask = df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])
    df = df[target_mask].copy()
    
    # Map Default to Charged Off for binary consistency if needed, or keeping robust
    df['loan_status'] = df['loan_status'].replace('Default', 'Charged Off')
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    print(f"Filtered Data (Fully Paid/Charged Off): {df.shape}")
    print(f"Default Rate: {df['target'].mean():.4f}")

    # Split
    print("Splitting data by time...")
    train_df, shift_df = split_data_by_time(df)
    print(f"Train Set (2014-2016): {train_df.shape}")
    print(f"Shift Set (2018-2019): {shift_df.shape}")
    
    # Visualizations
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    train_df['loan_status'].value_counts(normalize=True).plot(kind='bar', title='Train: Loan Status')
    plt.subplot(1, 2, 2)
    shift_df['loan_status'].value_counts(normalize=True).plot(kind='bar', title='Shift: Loan Status')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'))
    plt.close()
    
    # 2. Interest Rate Distribution
    plt.figure(figsize=(10, 5))
    sns.kdeplot(train_df['int_rate'], label='Train (2014-2016)', shade=True)
    sns.kdeplot(shift_df['int_rate'], label='Shift (2018-2019)', shade=True)
    plt.title('Distribution of Interest Rate')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'int_rate_distribution.png'))
    plt.close()
    
    # 3. FICO Score Distribution
    plt.figure(figsize=(10, 5))
    sns.kdeplot(train_df['fico_range_low'], label='Train', shade=True)
    sns.kdeplot(shift_df['fico_range_low'], label='Shift', shade=True)
    plt.title('Distribution of FICO Score (Low)')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'fico_distribution.png'))
    plt.close()

    print("EDA Plots saved.")

if __name__ == "__main__":
    perform_eda()
