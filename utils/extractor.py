"""
This script is to be placed in the same folder as the ZIP files you want to extract.
It will extract all ZIP files in the folder to a new folder with the same name as the ZIP file (without the .zip extension).
"""
import os
import zipfile

def extract_all_zips():
    # Get the current script directory
    folder_path = os.path.dirname(os.path.abspath(__file__))
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a ZIP file
        if filename.endswith('.zip'):
            file_path = os.path.join(folder_path, filename)
            # Create a folder with the same name as the ZIP file (without extension)
            extract_folder = os.path.join(folder_path, filename[:-4])
            os.makedirs(extract_folder, exist_ok=True)
            # Extract the ZIP file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f'Extracted {filename} to {extract_folder}')

# Call the function
extract_all_zips()
