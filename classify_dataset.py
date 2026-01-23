import os
import pandas as pd

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")

# The Logic: If a keyword (left) appears in the metadata, assign the Class (right)
# Priority: Top matches are checked first.
KEYWORD_RULES = {
    "Jewellery": [
        "amulet", "necklace", "ring", "bracelet", "earring", "pendant", "bead", 
        "scarab", "jewelry", "anklet", "collar", "pectoral", "diadem", "gold", "silver",
        "carnelian", "lapis", "faience inlay"
    ],
    "Statuary": [
        "statue", "statuette", "figure", "figurine", "sculpture", "head", "bust", 
        "sphinx", "torso", "ushabti", "shabti", "shawabty", "idol", "bronze"
    ],
    "Pottery": [
        "vessel", "jar", "pot", "cup", "bowl", "dish", "vase", "amphora", "plate", 
        "jug", "beaker", "flask", "canopic", "bottle", "ceramic", "terracotta", 
        "clay", "earthenware", "faience vessel", "pottery"
    ],
    "Reliefs": [
        "relief", "stela", "stele", "plaque", "frieze", "talatat", "wall fragment", 
        "block", "tomb relief", "painting", "ostracon", "ostraca", "fresco", "mural", 
        "limestone fragment"
    ]
}

def classify_text(text):
    if not isinstance(text, str):
        return None
    text = text.lower()
    
    for category, keywords in KEYWORD_RULES.items():
        for word in keywords:
            # We look for the word as a substring
            if word in text:
                return category
    return "Unclassified"

def process_csvs():
    # Get all CSV files
    files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    
    print(f"--- Auto-Classifying {len(files)} CSV Files ---")
    
    for filename in files:
        filepath = os.path.join(CSV_DIR, filename)
        df = pd.read_csv(filepath)
        
        # Only process if we have the necessary columns
        if 'Classification' not in df.columns:
            df['Classification'] = "Unknown"
            
        initial_unknowns = len(df[df['Classification'].isin(["Unknown", "Unclassified"])])
        
        # We combine text columns to search broadly
        # Title + ObjectName + Medium
        df['SearchText'] = (
            df['Title'].fillna('') + " " + 
            df['ObjectName'].fillna('') + " " + 
            df['Medium'].fillna('')
        )
        
        # Apply the logic
        # If it's ALREADY classified (e.g. from Chicago), we keep it.
        # We only update if it is "Unknown" or "Unclassified" or "nan".
        def apply_classification(row):
            current = str(row['Classification'])
            if current not in ["Unknown", "Unclassified", "nan"]:
                return current
            return classify_text(row['SearchText'])

        df['Classification'] = df.apply(apply_classification, axis=1)
        
        # Clean up
        df.drop(columns=['SearchText'], inplace=True)
        
        # Stats
        final_unknowns = len(df[df['Classification'] == "Unclassified"])
        fixed = initial_unknowns - final_unknowns
        
        # Save back to the SAME file
        df.to_csv(filepath, index=False)
        print(f"[{filename}] Fixed: {fixed} | Remaining Unknown: {final_unknowns}")

if __name__ == "__main__":
    process_csvs()