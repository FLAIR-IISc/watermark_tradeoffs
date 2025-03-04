# ==============================
# utils.py
# Description: Utility functions
# ==============================

import os
import json
import fcntl

def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None

def add_evaluation_record_to_file(file_path, record) -> None:
    """Add an evaluation record to a file."""

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(file_path):
        with open(file_path, "r+") as file: # add data to file if it exists
            fcntl.flock(file, fcntl.LOCK_EX) # lock file
            existing_list = json.load(file)
            existing_list.append(record)
            file.seek(0)
            json.dump(existing_list, file, indent=2)
            file.truncate()
            fcntl.flock(file, fcntl.LOCK_UN) # unlock file
    else:
        with open(file_path, "w") as file:
            fcntl.flock(file, fcntl.LOCK_EX) # lock file
            new_list = [record]
            json.dump(new_list, file, indent=2)
            fcntl.flock(file, fcntl.LOCK_UN) # unlock file