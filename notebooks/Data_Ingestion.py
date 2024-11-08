data = pd.read_csv('/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()

# Check for missing values
data.isnull().sum()

# Drop rows or impute missing values as appropriate
data.dropna(inplace=True)

