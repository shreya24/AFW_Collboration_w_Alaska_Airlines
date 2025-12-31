import os
import json
import csv
import glob

def extract_website(data):
    """Extract website URL from website search results"""
    if "organic" in data and len(data["organic"]) > 0:
        return data["organic"][0].get("link", "")
    return ""

def extract_address_from_snippet(data):
    """Extract address from organic snippet in address search results"""
    if "organic" in data and len(data["organic"]) > 0:
        return data["organic"][0].get("snippet", "")
    return ""

def extract_address_from_answerbox(data):
    """Extract address from answerBox in address search results"""
    if "answerBox" in data:
        if "answer" in data["answerBox"]:
            return data["answerBox"]["answer"]
        elif "snippet" in data["answerBox"]:
            return data["answerBox"]["snippet"]
    return ""

def extract_maps_address(data):
    """Extract address from maps search results"""
    if "places" in data and len(data["places"]) > 0:
        return data["places"][0].get("address", "")
    return ""

def process_agency_folder(folder_path):
    """Process a single agency folder and return extracted data"""
    agency_name = os.path.basename(folder_path)
    
    # Initialize data dictionary
    agency_data = {
        "agency_name": agency_name.replace("_", " "),
        "website": "",
        "address_from_search_snippet": "",
        "address_from_search_answerbox": "",
        "address_from_maps": ""
    }
    
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            file_name = os.path.basename(json_file)
            
            # Extract data based on file type
            if "search_website" in file_name:
                agency_data["website"] = extract_website(data)
            elif "search_address" in file_name:
                agency_data["address_from_search_snippet"] = extract_address_from_snippet(data)
                agency_data["address_from_search_answerbox"] = extract_address_from_answerbox(data)
            elif "maps" in file_name and "search" not in file_name:
                agency_data["address_from_maps"] = extract_maps_address(data)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return agency_data

def main():
    output_dir = "outputs_V2"
    csv_file = "agency_data_preprocessedV2_Peilu.csv"
    
    # Get all subdirectories in outputs (each is an agency)
    agency_folders = [f for f in glob.glob(os.path.join(output_dir, "*")) if os.path.isdir(f)]
    
    if not agency_folders:
        print("No agency folders found in the outputs directory.")
        return
    
    # Process each agency folder
    all_agency_data = []
    for folder in agency_folders:
        print(f"Processing: {folder}")
        agency_data = process_agency_folder(folder)
        all_agency_data.append(agency_data)
    
    # Write data to CSV
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ["agency_name", "website", "address_from_search_snippet", "address_from_search_answerbox", "address_from_maps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for agency_data in all_agency_data:
            writer.writerow(agency_data)
    
    print(f"Processing complete. Data saved to {csv_file}")

if __name__ == "__main__":
    main()
