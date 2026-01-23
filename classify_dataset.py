'''
I designed this script to act as a strict "Quality Control" manager for my dataset. 
Its job is to read the description of every artifact and decide exactly which category it belongs to, while following a very specific set of rules I created.

Why I Did It:

-I Focused on Shape, Not Material: I realized that classifying items based on what they are made of (like gold or stone) is not accurate. 
 A statue made of gold is still a statue, not jewelry. 
 So, I programmed this script to ignore materials and only look for specific object types, like "necklace," "amulet," or "figurine."

-I Removed Irrelevant History: My project is focused on Ancient Egypt. 
 I added a "Trash Can" list to automatically find and delete items that don't fit this timeline, such as European prints, Islamic manuscripts, or random tools.

-I Enforced Consistency: Instead of relying on the original museum labels, which might be vague, 
 I forced the script to re-evaluate every single item from scratch using my new, stricter rules.

The Result:
-This ensures my dataset is purely Ancient Egyptian and that every item is categorized by what it actually is, rather than what it is made of.
'''

import os
import pandas as pd

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")
CSV_FILES = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]

TRASH_KEYWORDS = [
    "flight into egypt", "print", "engraving", "woodcut", "lithograph", 
    "textile", "tunic", "garment", "sandal", "curtain", "carpet",
    "lance", "spear", "axe", "knife", "blade", "arrow", "tool",
    "quran", "koran", "manuscript", "folio", "calligraphy"
]

# CLASSIFICATION RULES
KEYWORD_RULES = {
    "Jewellery": [
        "amulet", "necklace", "ring", "bracelet", "earring", "pendant", "bead", 
        "scarab", "jewelry", "anklet", "collar", "pectoral", "diadem", 
        "seal", "finger ring" 
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
    print(f"--- Running The Strict Filter (Removing 'Gold' Bug) ---")
    
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
        def apply_logic(row):
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