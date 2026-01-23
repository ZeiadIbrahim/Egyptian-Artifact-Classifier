import os
import pandas as pd
from collections import Counter

CSV_DIR = os.path.join("dataset", "csv_files")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")

def inspect_unclassified():
    if not os.path.exists(MASTER_CSV):
        return

    df = pd.read_csv(MASTER_CSV)
    
    # Filter for Unclassified
    unknowns = df[df['Classification'].isin(['Unknown', 'Unclassified'])]
    
    print(f"--- Unclassified Detective ---")
    print(f"Total Unclassified Items: {len(unknowns)}")
    
    # Gather all text from Title and ObjectName
    text_blob = []
    for index, row in unknowns.iterrows():
        # Add Title and ObjectName to the blob, converting to string to avoid errors
        if pd.notna(row['Title']):
            text_blob.extend(str(row['Title']).lower().split())
        if pd.notna(row['ObjectName']):
            text_blob.extend(str(row['ObjectName']).lower().split())

    # Filter out boring words
    boring_words = {'the', 'of', 'and', 'a', 'in', 'with', 'from', 'fragment', 'unknown', 'egyptian', 'period'}
    clean_blob = [word.strip('.,()') for word in text_blob if word.strip('.,()') not in boring_words and len(word) > 3]
    
    # Count frequencies
    common_words = Counter(clean_blob).most_common(30)
    
    print("\nTop 30 Words:")
    for word, count in common_words:
        print(f"  {word}: {count}")

    print("\n--- Sample Entries ---")
    print(unknowns[['Title', 'ObjectName']].head(10).to_string(index=False))

if __name__ == "__main__":
    inspect_unclassified()