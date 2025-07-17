import pandas as pd
import os

# Path to MATRES folder with .txt files
MATRES_DIR = 'C:\\Users\\sohan\\Documents\\event-causal-temporal\\alldata\\MATRES\\'

# Load and combine TimeBank, AQUAINT, and Platinum datasets
def load_matres():
    files = ['TimeBank.txt', 'AQUAINT.txt', 'Platinum.txt']
    dfs = []
    for file in files:
        path = os.path.join(MATRES_DIR, file)
        df = pd.read_csv(path, sep='\t', header=None,
                         names=["doc_id", "e1_text", "e2_text","e1_id","e2_id", "relation"])
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)

# Load the dataset
df = load_matres()

# Preview the structure
print(df.head())

# Filter or map relation labels if needed
relation_map = {
    'BEFORE': 0,
    'AFTER': 1,
    'SIMULTANEOUS': 2,
    'VAGUE': 3
}
df['label'] = df['relation'].map(relation_map)

# Optional: Create a textual representation for modeling
def build_input(row):
    return f"[E1] {row['e1_text']} [/E1] ... [E2] {row['e2_text']} [/E2]"

df['input_text'] = df.apply(build_input, axis=1)

# Final dataset: input_text and label
preprocessed_df = df[['input_text', 'label']]
print(preprocessed_df.sample(5))
# Save the preprocessed dataset to a CSV file
preprocessed_df.to_csv('C:\\Users\\sohan\\Documents\\event-causal-temporal\\alldata\\MATRES\\MATRES_preprocessed.csv', index=False)
