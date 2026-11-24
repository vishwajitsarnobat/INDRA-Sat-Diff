import csv
import os
import shutil
import math # Used to round up to the nearest whole number

# --- 1. CONFIGURATION: PLEASE EDIT THESE THREE VARIABLES ---

# The full path to your CSV file that contains the filenames.
# Example for Windows: 'C:\\Users\\YourUser\\Documents\\my_files.csv'
# Example for macOS/Linux: '/home/youruser/documents/my_files.csv'
csv_file_path = '/home/sac/data_67/Nowcasting/data/input_sequences_south.csv'

# The path to the folder where all the source files are currently located.
source_folder = '/home/sac/data_67/Nowcasting/data/required_south'

# The path to the folder where you want to copy the selected files.
# The script will create this folder if it doesn't exist.
destination_folder = '/home/sac/data_67/Nowcasting/data/required_south_test'

# --- End of Configuration ---


def copy_file_subset():
    """
    Reads a CSV of filenames, and copies the top 1% of the
    sequences (rows) to a new directory.
    """
    print("--- Starting File Copy Process ---")

    # --- Step 1: Create the destination folder if it doesn't exist ---
    try:
        os.makedirs(destination_folder, exist_ok=True)
        print(f"Destination folder ready: {destination_folder}")
    except OSError as e:
        print(f"Error: Could not create destination folder. {e}")
        return

    # --- Step 2: Read all sequences (rows) from the CSV file ---
    try:
        with open(csv_file_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            # Read all rows into a list, preserving the order.
            # Filter out any completely empty rows.
            all_sequences = [row for row in reader if any(cell.strip() for cell in row)]
    except FileNotFoundError:
        print(f"Error: The CSV file was not found at '{csv_file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return
        
    total_sequences = len(all_sequences)
    if total_sequences == 0:
        print("The CSV file is empty. No files to copy.")
        return
        
    print(f"Found {total_sequences} sequences (rows) in the CSV file.")

    # --- Step 3: Calculate 1% and determine how many sequences to select ---
    # We use math.ceil to ensure that for small numbers (e.g., 50 rows), we still get at least 1.
    num_to_select = math.ceil(total_sequences * 0.1)
    num_to_select = int(num_to_select) # Convert to integer
    
    print(f"Calculating 1% of total: The top {num_to_select} sequence(s) will be selected for copying.")

    # --- Step 4: Select the TOP sequences from the list ---
    # We use list slicing to get the first 'num_to_select' items.
    selected_sequences = all_sequences[:num_to_select]

    # --- Step 5: Iterate through the selected sequences and copy the files ---
    files_copied_count = 0
    files_missing_count = 0
    
    print("\nStarting the copy process...")
    for i, sequence in enumerate(selected_sequences):
        print(f"  -> Processing sequence {i + 1}/{num_to_select} (from row {i + 1} of the CSV)...")
        for filename in sequence:
            filename = filename.strip() # Clean up any extra whitespace
            if not filename:
                continue # Skip if the cell in the CSV is empty

            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Check if the source file exists before trying to copy
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, destination_path)
                    files_copied_count += 1
                except Exception as e:
                    print(f"    - ERROR copying {filename}: {e}")
            else:
                print(f"    - WARNING: Source file not found, skipping: {source_path}")
                files_missing_count += 1

    # --- Final Summary ---
    print("\n--- Process Complete ---")
    print(f"Total sequences in CSV: {total_sequences}")
    print(f"Selected sequences from the top (1%): {num_to_select}")
    print(f"Successfully copied files: {files_copied_count}")
    if files_missing_count > 0:
        print(f"Files listed in CSV but not found in source folder: {files_missing_count}")
    print(f"All copied files are located in: {destination_folder}")


# Run the main function
if __name__ == "__main__":
    copy_file_subset()