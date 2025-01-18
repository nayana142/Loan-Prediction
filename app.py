import streamlit as st
import pandas as pd
import pickle


# Load the pre-trained model, encoder, and scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Application Title
st.title("Loan Approval Prediction App")
st.write("This app predicts whether a loan will be approved based on applicant details.")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Collect user inputs for all features
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, value=490000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
experience = st.sidebar.number_input("Work Experience (Years)", min_value=0, max_value=65, value=5)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value= 190000)
loan_duration = st.sidebar.number_input("Loan Duration (Months)", min_value=6, max_value=120, value=36)
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
number_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
home_ownership_status = st.sidebar.selectbox("Home Ownership Status", ["Own", "Mortgage", "Rent", "Other"])
monthly_debt_payments = st.sidebar.number_input("Monthly Debt Payments ($)", min_value=0, max_value=3000, value=500)
credit_card_utilization_rate = st.sidebar.number_input("Credit Card Utilization Rate (%)", min_value=0.0, max_value=1.0, value=0.3)
number_of_open_credit_lines = st.sidebar.number_input("Number of Open Credit Lines", min_value=0, max_value=15, value=5)
number_of_credit_inquiries = st.sidebar.number_input("Number of Credit Inquiries", min_value=0, max_value=10, value=2)
debt_to_income_ratio = st.sidebar.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=1.0, value=0.4)
bankruptcy_history = st.sidebar.selectbox("Bankruptcy History", [0, 1])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Debt Consolidation", "Education", "Other", "Auto"])
previous_loan_defaults = st.sidebar.selectbox("Previous Loan Defaults", [0, 1])
payment_history = st.sidebar.number_input("Payment History Score", min_value=0, max_value=100, value=80)
length_of_credit_history = st.sidebar.number_input("Length of Credit History (Years)", min_value=0, max_value=50, value=10)
savings_account_balance = st.sidebar.number_input("Savings Account Balance ($)", min_value=0, max_value=300089, value=5000)
checking_account_balance = st.sidebar.number_input("Checking Account Balance ($)", min_value=0,max_value=62572, value=2000)
total_assets = st.sidebar.number_input("Total Assets ($)", min_value=0,max_value=2619627, value=100000)
total_liabilities = st.sidebar.number_input("Total Liabilities ($)", min_value=0,max_value=1517302, value=50000)
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0,max_value=35000, value=2000)
utility_bills_payment_history = st.sidebar.number_input("Utility Bills Payment History Score", min_value=0.0, max_value=1.0, value=0.5)
job_tenure = st.sidebar.number_input("Job Tenure (Years)", min_value=0,max_value=20,value=5)
net_worth = st.sidebar.number_input("Net Worth ($)", min_value=-1000000,max_value=2603208, value=50000)
base_interest_rate = st.sidebar.number_input("Base Interest Rate (%)", min_value=0.0, max_value=1.0, value=0.5)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=1.0, value=0.5)
monthly_loan_payment = st.sidebar.number_input("Monthly Loan Payment ($)", min_value=0,max_value=20892, value=1000)
total_debt_to_income_ratio = st.sidebar.number_input("Total Debt to Income Ratio (%)", min_value=0, max_value=5, value=3)
risk_score = st.sidebar.number_input("Risk Score", min_value=0, max_value=100, value=60)

# Combine inputs into a dictionary
input_data = {
    "Age": [age],
    "AnnualIncome": [annual_income],
    "CreditScore": [credit_score],
    "EmploymentStatus": [employment_status],
    "EducationLevel": [education_level],
    "Experience": [experience],
    "LoanAmount": [loan_amount],
    "LoanDuration": [loan_duration],
    "MaritalStatus": [marital_status],
    "NumberOfDependents": [number_of_dependents],
    "HomeOwnershipStatus": [home_ownership_status],
    "MonthlyDebtPayments": [monthly_debt_payments],
    "CreditCardUtilizationRate": [credit_card_utilization_rate],
    "NumberOfOpenCreditLines": [number_of_open_credit_lines],
    "NumberOfCreditInquiries": [number_of_credit_inquiries],
    "DebtToIncomeRatio": [debt_to_income_ratio],
    "BankruptcyHistory": [bankruptcy_history],
    "LoanPurpose": [loan_purpose],
    "PreviousLoanDefaults": [previous_loan_defaults],
    "PaymentHistory": [payment_history],
    "LengthOfCreditHistory": [length_of_credit_history],
    "SavingsAccountBalance": [savings_account_balance],
    "CheckingAccountBalance": [checking_account_balance],
    "TotalAssets": [total_assets],
    "TotalLiabilities": [total_liabilities],
    "MonthlyIncome": [monthly_income],
    "UtilityBillsPaymentHistory": [utility_bills_payment_history],
    "JobTenure": [job_tenure],
    "NetWorth": [net_worth],
    "BaseInterestRate": [base_interest_rate],
    "InterestRate": [interest_rate],
    "MonthlyLoanPayment": [monthly_loan_payment],
    "TotalDebtToIncomeRatio": [total_debt_to_income_ratio],
    "RiskScore": [risk_score],
}

# Convert to DataFrame
new_data = pd.DataFrame(input_data)

# Encode categorical features
categorical_features = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]
encoded_df = encoder.transform(new_data[categorical_features])
encoded_data = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(categorical_features))

# Combine with numerical features
new_data = new_data.drop(columns=categorical_features)
new_data = pd.concat([new_data, encoded_data], axis=1)

# Scale data
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

# Center the predict button
st.markdown("---")  # Add a divider
col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns
with col2:  # Place the button in the center column
    predict_clicked = st.button("Predict")  # Button click action

# Show the prediction result on the next line, left-aligned
if predict_clicked:
    prediction = model.predict(new_data_scaled)  # Perform prediction
    st.markdown("### Prediction Result")  # Section header
    if prediction[0] == 1:
        st.success("Loan Approved!", icon="✅")  # Success message
    else:
        st.error("Loan Rejected.", icon="❌")  