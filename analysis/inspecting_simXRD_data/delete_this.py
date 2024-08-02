from ase.db import connect
from tqdm import tqdm
import random

# Connect to the database
test_full_data_path = "ML_For_XRD_Materials_Characterisation/training_data/simXRD_full_data/test.db"
test_partial_data_path = "ML_For_XRD_Materials_Characterisation/training_data/simXRD_partial_data/test.db"
databs = connect(test_partial_data_path)

def safe_print_db_info(db, sample_size=10):
    total_entries = len(db)
    print(f"Total number of entries: {total_entries}")
    
    # Sample random entries
    sample_ids = random.sample(range(1, total_entries + 1), min(sample_size, total_entries))
    
    print(f"\nAnalyzing a sample of {len(sample_ids)} entries:")
    
    all_keys = set()
    all_data_keys = set()
    
    for id in tqdm(sample_ids, desc="Processing entries"):
        entry = db.get(id=id)
        all_keys.update(entry._keys)
        if hasattr(entry, 'data'):
            all_data_keys.update(entry.data.keys())
    
    print("\nAttributes found across sampled entries:")
    for key in sorted(all_keys):
        print(f"  {key}")
    
    print("\nKeys in .data attribute found across sampled entries:")
    for key in sorted(all_data_keys):
        print(f"  {key}")

if __name__ == "__main__":
    safe_print_db_info(databs)