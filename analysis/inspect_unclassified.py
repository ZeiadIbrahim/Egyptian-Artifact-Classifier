"""
-I built a thisscript to investigate the mystery items in my dataset. 
 I had hundreds of artifacts labeled "Unknown," and instead of guessing or checking them one by one, I wrote code to read and analyze their descriptions for me.

Why I Did It:

-I Needed to Find Patterns: I couldn't classify these items because their labels were missing. 
 By counting the most common words in their titles (like "coffin," "model," or "print"), I could figure out what groups they actually belonged to.

-I Filtered Out Noise: To make the results useful, I programmed the script to ignore boring, common words like "the," "and," or "Egyptian." 
 This ensured that only the important keywords—the ones that actually describe the object—rose to the top.

-I Needed Evidence: I didn't want to make random guesses. 
 This script provided the hard data I needed to decide which "Unknown" items were valuable artifacts (to be saved) and which were irrelevant (to be deleted).

-The Result: This analysis gave me a clear 30 word list of keywords, 
 which revealed exactly how to fix the gaps in my dataset and re-classify the unknown items correctly.
"""

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