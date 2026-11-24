import csv
import os
import shutil
from tqdm import tqdm # Optional: for progress bar

def copy_unique_files_from_csv(csv_file_path, input_dir_path, output_dir_path):
    """
    Reads filenames from a CSV, finds unique ones present in the input directory,
    and copies them to the output directory.

    Args:
        csv_file_path (str): Path to the CSV file.
        input_dir_path (str): Path to the directory containing the source files.
        output_dir_path (str): Path to the directory where unique files will be copied.
    """

    # --- 1. Input Validation ---
    if not os.path.isfile(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return
    if not os.path.isdir(input_dir_path):
        print(f"Error: Input directory not found at '{input_dir_path}'")
        return

    # --- 2. Create Output Directory (if it doesn't exist) ---
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output directory: '{output_dir_path}'")
    except OSError as e:
        print(f"Error creating output directory '{output_dir_path}': {e}")
        return

    # --- 3. Read Filenames from CSV ---
    all_filenames_in_csv = set()
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            print(f"Reading filenames from '{csv_file_path}'...")
            for row in reader:
                for filename in row:
                    cleaned_filename = filename.strip() # Remove leading/trailing whitespace
                    if cleaned_filename: # Avoid empty strings
                        all_filenames_in_csv.add(cleaned_filename)
    except FileNotFoundError:
        print(f"Error: Could not open CSV file '{csv_file_path}'") # Should be caught by initial check, but good practice
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not all_filenames_in_csv:
        print("No filenames found in the CSV file.")
        return

    print(f"Found {len(all_filenames_in_csv)} unique filenames listed in the CSV.")

    # --- 4. Identify Files to Copy and Perform Copying ---
    copied_count = 0
    missing_count = 0
    skipped_count = 0
    print(f"Checking files in '{input_dir_path}' and copying to '{output_dir_path}'...")

    # Use tqdm for a progress bar (optional)
    for filename in tqdm(all_filenames_in_csv, desc="Copying files"):
        source_path = os.path.join(input_dir_path, filename)
        destination_path = os.path.join(output_dir_path, filename)

        if os.path.isfile(source_path):
            # Optional: Check if file already exists in destination to avoid re-copying
            if os.path.exists(destination_path):
                 # print(f"Skipping: '{filename}' already exists in destination.")
                 skipped_count += 1
                 continue # Skip if it already exists

            try:
                shutil.copy2(source_path, destination_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"\nError copying file '{filename}': {e}") # Print error with newline if using tqdm
                missing_count += 1 # Count as missing if copy fails
        else:
            # print(f"Warning: File '{filename}' listed in CSV but not found in '{input_dir_path}'")
            missing_count += 1

    # --- 5. Print Summary ---
    print("\n--- Copy Summary ---")
    print(f"Unique filenames listed in CSV: {len(all_filenames_in_csv)}")
    print(f"Files found and copied:         {copied_count}")
    print(f"Files skipped (already exist):  {skipped_count}")
    print(f"Files listed but not found:     {missing_count}")
    print("--------------------")

if __name__ == "__main__":

    csv_file = "src/prediff/datasets/imerg/input_sequences_south.csv"
    input_dir = "data/processed_south"
    output_dir = "data/required_south"
    copy_unique_files_from_csv(csv_file, input_dir, output_dir)