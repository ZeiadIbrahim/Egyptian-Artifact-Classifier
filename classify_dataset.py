import os
import pandas as pd

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")
CSV_FILES = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]

# 1. If these words appear, we mark the row for DELETION.
# (European prints, random tools, textiles that confuse the AI)
TRASH_KEYWORDS = [
    "flight into egypt", "print", "engraving", "woodcut", "lithograph", 
    "textile", "tunic", "garment", "sandal", "curtain", "carpet",
    "lance", "spear", "axe", "knife", "blade", "arrow", "tool"
]

# 2. CLASSIFICATION RULES (Updated with your findings)
KEYWORD_RULES = {
    "Jewellery": [
        "amulet", "necklace", "ring", "bracelet", "earring", "pendant", "bead", 
        "scarab", "jewelry", "anklet", "collar", "pectoral", "diadem", "gold", "silver",
        "carnelian", "lapis", "faience inlay", "seal", "finger ring"
    ],
    "Statuary": [
        "statue", "statuette", "figure", "figurine", "sculpture", "head", "bust", 
        "sphinx", "torso", "ushabti", "shabti", "shawabty", "idol", "bronze",
        "coffin", "sarcophagus", "mask", "cartonnage", "mummy case", "model", "boat"
    ],
    "Pottery": [
        "vessel", "jar", "pot", "cup", "bowl", "dish", "vase", "amphora", "plate", 
        "jug", "beaker", "flask", "canopic", "bottle", "ceramic", "terracotta", 
        "clay", "earthenware", "faience vessel", "pottery", "chalice"
    ],
    "Reliefs": [
        "relief", "stela", "stele", "plaque", "frieze", "talatat", "wall fragment", 
        "block", "tomb relief", "painting", "ostracon", "ostraca", "fresco", "mural", 
        "limestone fragment", "facsimile", "papyrus", "drawing", "copy"
    ]
}

def get_classification(text):
    if not isinstance(text, str): return "Unclassified"
    text = text.lower()
    
    # Check for Trash first
    for bad_word in TRASH_KEYWORDS:
        if bad_word in text:
            return "DELETE"
            
    # Check for Good Classes
    for category, keywords in KEYWORD_RULES.items():
        for word in keywords:
            if word in text:
                return category
    return "Unclassified"

def process_csvs():
    print(f"--- Running The Great Filter ---")
    
    for filename in CSV_FILES:
        filepath = os.path.join(CSV_DIR, filename)
        df = pd.read_csv(filepath)
        
        # Combine text for searching
        df['SearchText'] = (
            df['Title'].fillna('') + " " + 
            df['ObjectName'].fillna('') + " " + 
            df['Medium'].fillna('')
        )
        
        # Apply the new logic
        # We RE-EVALUATE "Unknown", "Unclassified", OR anything that looks like it might be trash
        def apply_logic(row):
            current = str(row['Classification'])
            # If it's already a solid class, we double check it isn't trash
            if current in KEYWORD_RULES.keys():
                for bad in TRASH_KEYWORDS:
                    if bad in row['SearchText'].lower():
                        return "DELETE"
                return current
            
            # If it's unknown, we try to classify it
            return get_classification(row['SearchText'])

        df['Classification'] = df.apply(apply_logic, axis=1)
        
        # REMOVE the rows marked DELETE
        initial_count = len(df)
        df = df[df['Classification'] != "DELETE"]
        deleted_count = initial_count - len(df)
        
        # Save
        df.drop(columns=['SearchText'], inplace=True)
        df.to_csv(filepath, index=False)
        
        print(f"[{filename}]")
        print(f"  - Purged (Trash): {deleted_count}")
        print(f"  - Remaining:      {len(df)}")

if __name__ == "__main__":
    process_csvs()