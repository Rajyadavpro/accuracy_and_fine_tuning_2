import os
import json
import fnmatch

def should_ignore(name, ignore_patterns):
    """
    Checks if a file or directory name matches any of the ignore patterns.
    """
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False

def folder_to_dict(dir_path, ignore_patterns=None):
    """
    Recursively builds a dictionary representing the folder structure.
    Files contain their text content, and directories contain nested dictionaries.
    """
    if ignore_patterns is None:
        ignore_patterns = []

    result = {}
    
    try:
        items = os.listdir(dir_path)
    except OSError as e:
        # Handle cases where folder cannot be read (e.g., permission issues)
        print(f"Warning: Could not read directory {dir_path}. Error: {e}")
        return result

    for item in items:
        # Skip items that match the ignore patterns
        if should_ignore(item, ignore_patterns):
            continue
        
        full_path = os.path.join(dir_path, item)
        
        if os.path.isdir(full_path):
            # Recurse into the subdirectory
            result[item] = folder_to_dict(full_path, ignore_patterns)
        elif os.path.isfile(full_path):
            try:
                # Read the file content. 
                # 'errors="replace"' is used to handle files with non-UTF-8 characters gracefully.
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    result[item] = f.read()
            except Exception as e:
                result[item] = f"<Error reading file: {e}>"
                
    return result

if __name__ == "__main__":
    # --- Configuration ---
    # Replace this with the path to the directory you want to scan
    target_directory = "."  
    
    # Define files, folders, or patterns you want to ignore
    ignore_list = [
        ".git",
        "node_modules",
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        "*.png",       # Ignores binary image files
        "*.jpg",
        "*.zip",
        "local.settings.json",  
        ".venv",
        "AUX_code",
        "combine.py",
        "project_structure.json"  # Ignore the output JSON file itself
    ]
    
    # Path where you want to save the final JSON file
    output_json_path = "project_structure.json"
    # ----------------------

    if os.path.exists(target_directory):
        print(f"Scanning directory: {target_directory}...")
        
        # Generate the dictionary
        folder_dict = folder_to_dict(target_directory, ignore_list)
        
        # Convert the dictionary to a pretty-printed JSON string
        json_data = json.dumps(folder_dict, indent=4, ensure_ascii=False)
        
        # Write JSON to a file
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_data)
            
        print(f"JSON representation successfully saved to: {output_json_path}")
    else:
        print(f"Error: The target directory '{target_directory}' does not exist.")