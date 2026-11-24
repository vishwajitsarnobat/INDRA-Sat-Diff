import os
import sys # Import sys to redirect stdout

def print_directory_tree(startpath, output_file_path=None, indent_char='|   ', file_prefix='├── ', dir_prefix='└── '):
    """
    Prints the directory structure in a tree-like format, either to console or to a file.

    Args:
        startpath (str): The path to the directory from which to start the tree.
        output_file_path (str, optional): The path to the file where the output should be saved.
                                          If None, output is printed to the console.
        indent_char (str): The characters used for indentation at each level.
                           Defaults to '|   ' for a classic tree look.
        file_prefix (str): Prefix for file entries. Defaults to '├── '.
        dir_prefix (str): Prefix for directory entries. Defaults to '└── '.
    """
    if not os.path.exists(startpath):
        print(f"Error: Path '{startpath}' does not exist.")
        return
    if not os.path.isdir(startpath):
        print(f"Error: Path '{startpath}' is not a directory.")
        return

    # Determine where to write the output
    if output_file_path:
        # Open the file in write mode, creating/overwriting it
        # Use 'with' statement for proper file handling (auto-closes)
        output_stream = open(output_file_path, 'w', encoding='utf-8')
        print(f"Saving directory tree to: {output_file_path}")
    else:
        # If no file path, print to standard output (console)
        output_stream = sys.stdout

    try:
        # Print the root directory name
        output_stream.write(f"{os.path.basename(os.path.abspath(startpath))}\n")

        # os.walk yields (dirpath, dirnames, filenames) for each directory it visits
        for root, dirs, files in os.walk(startpath):
            # Calculate the current level of indentation
            # The level is determined by how many subdirectories deep 'root' is from 'startpath'
            # Ensure startpath ends with os.sep for consistent counting, especially for root directory itself
            relative_path = os.path.relpath(root, startpath)
            if relative_path == ".": # Root directory itself
                level = 0
            else:
                level = relative_path.count(os.sep) + 1 # +1 because os.relpath for 'dir/subdir' is 'subdir' (0 sep)

            indent = indent_char * level

            # Print directories at the current level
            for d in sorted(dirs):
                output_stream.write(f"{indent}{dir_prefix}{d}{os.sep}\n")

            # Print files at the current level
            for f in sorted(files):
                filename, file_extension = os.path.splitext(f)
                if file_extension:
                    output_stream.write(f"{indent}{file_prefix}{f} (Ext: {file_extension})\n")
                else:
                    output_stream.write(f"{indent}{file_prefix}{f}\n")

    finally:
        # Close the file if it was opened by this function
        if output_file_path:
            output_stream.close()

# --- Example Usage ---
if __name__ == "__main__":
    # Get the current working directory as the default starting path
    current_directory = os.getcwd()
    
    # Define the output file path
    output_file = "directory_tree.txt"

    print(f"Scanning directory: {current_directory}")
    print(f"Output will be saved to: {output_file}\n")
    
    # Call the function to save to file
    print_directory_tree(current_directory, output_file_path=output_file)

    print(f"\nDirectory tree successfully saved to '{output_file}'.")

    # You can also test printing to console by not providing output_file_path
    # print("\n--- Testing console output (for a small part of the tree) ---")
    # print_directory_tree(os.path.dirname(current_directory), output_file_path=None) # Example: parent directory to avoid too much console output
