import pandas as pd
import numpy as np

# LOAD
df = pd.read_csv("Clinical_Trial_module/Data/Clinical_processed/nlp_enhanced_data.csv")

# TARGET
status_col = [col for col in df.columns if 'status' in col][0]
df['target'] = df[status_col].apply(lambda x: 1 if str(x).lower() == 'completed' else 0)
df = df.drop(columns=[status_col])

# DROP USELESS
df = df.drop(columns=['nct_number', 'start_date', 'primary_completion_date', 'completion_date'], errors='ignore')

# FEATURE / TARGET
X = df.drop(columns=['target'])
X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.astype(str)
X.columns = X.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

# Print all columns to a file
with open("all_cols.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("Total Features found:", len(X.columns))
