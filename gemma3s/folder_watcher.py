import json
import time
import argparse
import os
from pathlib import Path
from datetime import datetime

# Configuration
# BASE_DIR is the directory where this script is located (MedSAM2 root)
BASE_DIR = Path(__file__).resolve().parent
COMMON_DATA_DIR = BASE_DIR / "../common_data"
JSON_FILE = COMMON_DATA_DIR / "comman_format.json"

def load_json(filepath):
    """Load JSON data safely, handling empty or missing files."""
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: {filepath} is not a list. Resetting to empty list.")
                return []
            return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON {filepath}. Treating as empty.")
        return []

def save_json(filepath, data):
    """Save JSON data atomically to avoid corruption."""
    temp_path = filepath.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=4)
        temp_path.replace(filepath)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updated {filepath}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        if temp_path.exists():
            temp_path.unlink()

def scan_and_update():
    """Scan for new patient folders and update the JSON registry."""
    if not COMMON_DATA_DIR.exists():
        print(f"Directory {COMMON_DATA_DIR} does not exist.")
        return

    current_data = load_json(JSON_FILE)
    existing_pids = {entry.get("pid") for entry in current_data}
    
    # List all pid_XXX directories
    patient_dirs = [d for d in COMMON_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("pid_")]
    
    new_entries = []
    
    for p_dir in patient_dirs:
        pid = p_dir.name
        
        if pid not in existing_pids:
            print(f"Found new patient: {pid}")
            
            # Optionally check for sessions, though not strictly required for the entry creation based on requirements
            # But let's see if we can find any session info to add context if needed.
            # Requirement says: "Keep the tumor key for the patient as None for new entry."
            
            new_entry = {
                "pid": pid,
                "tumor": None,
                "reviewed_by_radio": False,
                "gemma_hard_coded_remark": None,
                 # We can initialize other fields if necessary, but these form the core requirement
            }
            new_entries.append(new_entry)
            existing_pids.add(pid) # Prevents duplicates if we loop or logic changes in future
            
    if new_entries:
        current_data.extend(new_entries)
        save_json(JSON_FILE, current_data)
    else:
        # No changes
        pass

def main():
    parser = argparse.ArgumentParser(description="Watch common_data for new patient folders.")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds (default: 10)")
    args = parser.parse_args()

    print(f"Starting Folder Watcher on {COMMON_DATA_DIR}")
    print(f"Polling every {args.interval} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            scan_and_update()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping Folder Watcher.")

if __name__ == "__main__":
    main()
