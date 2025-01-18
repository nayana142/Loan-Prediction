import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

# Number of samples
num_samples = 2000

# Seed for reproducibility
np.random.seed(42)

def generate_correlated_features(num_samples):
    # Generate base features
    age = np.random.normal(40, 12, num_samples).clip(18, 80).astype(int)
    experience = (age - 18 - np.random.normal(4, 2, num_samples).clip(0)).clip(0).astype(int)
    education_level = np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], num_samples, p=[0.3, 0.2, 0.3, 0.15, 0.05])
    
    # Education affects income and credit score
    edu_impact = {'High School': 0, 'Associate': 0.1, 'Bachelor': 0.2, 'Master': 0.3, 'Doctorate': 0.4}
    edu_factor = np.array([edu_impact[level] for level in education_level])
    
    # Generate correlated income, credit score, and employment status
    base_income = np.random.lognormal(10.5, 0.6, num_samples) * (1 + edu_factor) * (1 + experience / 100)
    income_noise = np.random.normal(0, 0.1, num_samples)
    annual_income = (base_income * (1 + income_noise)).clip(15000, 300000).astype(int)
    
    credit_score_base = 300 + 300 * stats.beta.rvs(5, 1.5, size=num_samples)
    credit_score = (credit_score_base + edu_factor * 100 + experience * 1.5 + income_noise * 100).clip(300, 850).astype(int)
    
    employment_status_probs = np.column_stack([
        0.9 - edu_factor * 0.3,  # Employed
        0.05 + edu_factor * 0.2,  # Self-Employed
        0.05 + edu_factor * 0.1   # Unemployed
    ])
    employment_status = np.array(['Employed', 'Self-Employed', 'Unemployed'])[np.argmax(np.random.random(num_samples)[:, np.newaxis] < employment_status_probs.cumsum(axis=1), axis=1)]
    
    return age, experience, education_level, annual_income, credit_score, employment_status

def generate_time_based_features(num_samples):
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    return dates

age, experience, education_level, annual_income, credit_score, employment_status = generate_correlated_features(num_samples)
application_dates = generate_time_based_features(num_samples)

data = {
    'ApplicationDate': application_dates,
    'Age': age,
    'AnnualIncome': annual_income,
    'CreditScore': credit_score,
    'EmploymentStatus': employment_status,
    'EducationLevel': education_level,
    'Experience': experience,
    'LoanAmount': np.random.lognormal(10, 0.5, num_samples).astype(int),
    'LoanDuration': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], num_samples, p=[0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025]),
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_samples, p=[0.3, 0.5, 0.15, 0.05]),
    'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4, 5], num_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
    'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], num_samples, p=[0.2, 0.3, 0.4, 0.1]),
    'MonthlyDebtPayments': np.random.lognormal(6, 0.5, num_samples).astype(int),
    'CreditCardUtilizationRate': np.random.beta(2, 5, num_samples),
    'NumberOfOpenCreditLines': np.random.poisson(3, num_samples).clip(0, 15).astype(int),
    'NumberOfCreditInquiries': np.random.poisson(1, num_samples).clip(0, 10).astype(int),
    'DebtToIncomeRatio': np.random.beta(2, 5, num_samples),
    'BankruptcyHistory': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
    'LoanPurpose': np.random.choice(['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'], num_samples, p=[0.3, 0.2, 0.15, 0.25, 0.1]),
    'PreviousLoanDefaults': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
    'PaymentHistory': np.random.poisson(24, num_samples).clip(0, 60).astype(int),
    'LengthOfCreditHistory': np.random.randint(1, 30, num_samples),
    'SavingsAccountBalance': np.random.lognormal(8, 1, num_samples).astype(int),
    'CheckingAccountBalance': np.random.lognormal(7, 1, num_samples).astype(int),
    'TotalAssets': np.random.lognormal(11, 1, num_samples).astype(int),
    'TotalLiabilities': np.random.lognormal(10, 1, num_samples).astype(int),
    'MonthlyIncome': annual_income / 12,
    'UtilityBillsPaymentHistory': np.random.beta(8, 2, num_samples),
    'JobTenure': np.random.poisson(5, num_samples).clip(0, 40).astype(int),
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure TotalAssets is always greater than or equal to the sum of SavingsAccountBalance and CheckingAccountBalance
df['TotalAssets'] = np.maximum(df['TotalAssets'], df['SavingsAccountBalance'] + df['CheckingAccountBalance'])

# Add more complex derived features
min_net_worth = 1000  # Set a minimum net worth
df['NetWorth'] = np.maximum(df['TotalAssets'] - df['TotalLiabilities'], min_net_worth)

# More realistic interest rate based on credit score, loan amount, and loan duration
df['BaseInterestRate'] = 0.03 + (850 - df['CreditScore']) / 2000 + df['LoanAmount'] / 1000000 + df['LoanDuration'] / 1200
df['InterestRate'] = df['BaseInterestRate'] * (1 + np.random.normal(0, 0.1, num_samples)).clip(0.8, 1.2)

df['MonthlyLoanPayment'] = (df['LoanAmount'] * (df['InterestRate']/12)) / (1 - (1 + df['InterestRate']/12)**(-df['LoanDuration']))
df['TotalDebtToIncomeRatio'] = (df['MonthlyDebtPayments'] + df['MonthlyLoanPayment']) / df['MonthlyIncome']

# Create a more complex loan approval rule
def loan_approval_rule(row):
    score = 0
    score += (row['CreditScore'] - 600) / 250  # Credit score factor
    score += (100000 - row['AnnualIncome']) / 100000  # Income factor
    score += (row['TotalDebtToIncomeRatio'] - 0.4) * 2  # DTI factor
    score += (row['LoanAmount'] - 10000) / 90000  # Loan amount factor
    score += (row['InterestRate'] - 0.05) * 10  # Interest rate factor
    score += 0.5 if row['BankruptcyHistory'] == 1 else 0  # Bankruptcy penalty
    score += 0.3 if row['PreviousLoanDefaults'] == 1 else 0  # Previous default penalty
    score += 0.2 if row['EmploymentStatus'] == 'Unemployed' else 0  # Employment status factor
    score -= 0.1 if row['HomeOwnershipStatus'] in ['Own', 'Mortgage'] else 0  # Home ownership factor
    score -= row['PaymentHistory'] / 120  # Payment history factor
    score -= row['LengthOfCreditHistory'] / 60  # Length of credit history factor
    score -= row['NetWorth'] / 500000  # Net worth factor
    
    # Age factor (slight preference for middle-aged applicants)
    score += abs(row['Age'] - 40) / 100
    
    # Experience factor
    score -= row['Experience'] / 200
    
    # Education factor
    edu_score = {'High School': 0.2, 'Associate': 0.1, 'Bachelor': 0, 'Master': -0.1, 'Doctorate': -0.2}
    score += edu_score[row['EducationLevel']]
    
    # Seasonal factor (higher approval rates in spring/summer)
    month = row['ApplicationDate'].month
    score -= 0.1 if 3 <= month <= 8 else 0
    
    # Random factor to add some unpredictability
    score += np.random.normal(0, 0.1)
    
    return 1 if score < 1 else 0  # Adjust this threshold to change overall approval rate

df['LoanApproved'] = df.apply(loan_approval_rule, axis=1)

# Add some noise and outliers
noise_mask = np.random.choice([True, False], num_samples, p=[0.01, 0.99])
df.loc[noise_mask, 'AnnualIncome'] = (df.loc[noise_mask, 'AnnualIncome'] * np.random.uniform(1.5, 2.0, noise_mask.sum())).astype(int)

low_net_worth_mask = df['NetWorth'] == min_net_worth
df.loc[low_net_worth_mask, 'NetWorth'] += np.random.randint(0, 10000, size=low_net_worth_mask.sum())

# Print some statistics
print(f"Loan Approval Rate: {df['LoanApproved'].mean():.2%}")
print(f"Average Credit Score: {df['CreditScore'].mean():.0f}")
print(f"Average Annual Income: ${df['AnnualIncome'].mean():.0f}")
print(f"Average Loan Amount: ${df['LoanAmount'].mean():.0f}")
print(f"Average Total Debt-to-Income Ratio: {df['TotalDebtToIncomeRatio'].mean():.2f}")
print(f"Average Interest Rate: {df['InterestRate'].mean():.2%}")

def assign_credit_score_risk(credit_score):
    if credit_score >= 750: return 1
    elif 700 <= credit_score < 750: return 2
    elif 650 <= credit_score < 700: return 3
    elif 600 <= credit_score < 650: return 4
    else: return 5

def assign_dti_risk(dti):
    if dti < 0.20: return 1
    elif 0.20 <= dti < 0.30: return 2
    elif 0.30 <= dti < 0.40: return 3
    elif 0.40 <= dti < 0.50: return 4
    else: return 5

def assign_payment_history_risk(payment_history):
    if payment_history >= 99: return 1
    elif 97 <= payment_history < 99: return 2
    elif 95 <= payment_history < 97: return 3
    elif 90 <= payment_history < 95: return 4
    else: return 5

def assign_bankruptcy_risk(bankruptcy_history):
    return 5 if bankruptcy_history else 1

def assign_previous_defaults_risk(previous_defaults):
    if previous_defaults == 0: return 1
    elif previous_defaults == 1: return 3
    else: return 5

def assign_utilization_risk(utilization):
    if utilization < 0.20: return 1
    elif 0.20 <= utilization < 0.40: return 2
    elif 0.40 <= utilization < 0.60: return 3
    elif 0.60 <= utilization < 0.80: return 4
    else: return 5

def assign_credit_history_risk(length_of_history):
    if length_of_history >= 10: return 1
    elif 7 <= length_of_history < 10: return 2
    elif 5 <= length_of_history < 7: return 3
    elif 3 <= length_of_history < 5: return 4
    else: return 5

def assign_income_risk(annual_income):
    if annual_income >= 120000: return 1
    elif 80000 <= annual_income < 120000: return 2
    elif 50000 <= annual_income < 80000: return 3
    elif 30000 <= annual_income < 50000: return 4
    else: return 5

def assign_employment_risk(employment_status):
    if employment_status == 'Employed': return 1
    elif employment_status == 'Self-employed': return 2
    elif employment_status == 'Part-time': return 3
    else: return 4  # Unemployed or other

def assign_net_worth_risk(net_worth):
    if net_worth >= 500000: return 1
    elif 250000 <= net_worth < 500000: return 2
    elif 100000 <= net_worth < 250000: return 3
    elif 50000 <= net_worth < 100000: return 4
    else: return 5

# Refined overall risk calculation
def calculate_overall_risk(row):
    base_score = (
        assign_credit_score_risk(row['CreditScore']) * 3 +
        assign_dti_risk(row['DebtToIncomeRatio']) * 2 +
        assign_payment_history_risk(row['PaymentHistory']) * 2 +
        assign_bankruptcy_risk(row['BankruptcyHistory']) * 3 +
        assign_previous_defaults_risk(row['PreviousLoanDefaults']) * 3 +
        assign_utilization_risk(row['CreditCardUtilizationRate']) +
        assign_credit_history_risk(row['LengthOfCreditHistory']) +
        assign_income_risk(row['AnnualIncome']) +
        assign_employment_risk(row['EmploymentStatus']) +
        assign_net_worth_risk(row['NetWorth']) * 2
    )
    
    # Adjust score based on loan approval status
    if row['LoanApproved'] == 1:  # Assuming 1 means approved
        base_score *= 0.8  # Reduce risk score for approved loans
    
    return base_score

# Apply the refined risk calculation
df['RiskScore'] = df.apply(calculate_overall_risk, axis=1)

# Save to CSV
df.to_csv('focused_synthetic_loan_data.csv', index=False)
print("\nFocused synthetic data saved to 'focused_synthetic_loan_data.csv'")

# Display final feature count
print(f"\nTotal number of features (including label): {len(df.columns)}")
print("\nFeatures:")
for column in df.columns:
    print(f"- {column}")