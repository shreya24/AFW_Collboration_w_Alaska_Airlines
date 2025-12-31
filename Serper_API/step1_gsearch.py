import requests
import json
import os
import csv
import time
import re
from datetime import datetime
import concurrent.futures
import tqdm
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Serper API key from .env
API_KEY = os.getenv("SERPER_API_KEY")

# Control concurrency settings from .env
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 30))  # Reduced from 100 to prevent connection exhaustion
MIN_REQUEST_DELAY = float(os.getenv("MIN_REQUEST_DELAY", 0.1))  # Minimum delay between requests in seconds
MAX_REQUEST_DELAY = float(os.getenv("MAX_REQUEST_DELAY", 0.3))  # Maximum delay for jitter
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 300))  # Process agencies in batches to release resources
BATCH_PAUSE = int(os.getenv("BATCH_PAUSE", 1))  # Seconds to pause between batches

# Create a session with retry configuration
def create_session():
    session = requests.Session()
    
    # Configure retry strategy with exponential backoff
    retries = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["POST", "GET"],  # HTTP methods to retry
    )
    
    # Apply retry strategy to both http and https
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def search_serper(query, api_key, search_type="search", retries=3):
    """
    Search using Serper API for the specific query
    search_type can be "search" or "maps"
    """
    # Determine the endpoint based on the search type
    if search_type == "maps":
        endpoint = "https://google.serper.dev/maps"
    else:
        endpoint = "https://google.serper.dev/search"
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    search_params = {
        "q": query,
        "gl": "us",
        "hl": "en"
    }
    
    # For regular search, we want to limit results
    if search_type == "search":
        search_params["num"] = 5
    
    # Create a dedicated session for this request
    session = create_session()
    
    try:
        # Add randomized delay to avoid hitting rate limits and predictable patterns
        jitter = random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)
        time.sleep(jitter)
        
        response = session.post(endpoint, headers=headers, json=search_params, timeout=(10, 30))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        # If we still have retries left and it's a connection error, try again with longer delay
        if retries > 0 and ("Connection" in str(e) or "Timeout" in str(e) or "Max retries" in str(e)):
            print(f"Retrying search for '{query}' ({retries} retries left)")
            # Exponential backoff
            time.sleep(2 ** (4 - retries) + random.uniform(1, 3))
            return search_serper(query, api_key, search_type, retries - 1)
        return None
    finally:
        # Close the session to release resources
        session.close()

def sanitize_folder_name(name):
    """
    Create a safe folder name from the agency name
    """
    # Replace invalid characters with underscores
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Replace spaces with underscores
    safe_name = safe_name.replace(" ", "_")
    # Replace multiple underscores with a single one
    safe_name = re.sub(r'_+', "_", safe_name)
    return safe_name

def save_response(response_data, agency_name, search_type, query_suffix=""):
    """
    Save the response to a file in the agency-specific directory
    """
    # Create base outputs directory
    base_dir = "outputs_V2"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Sanitize agency name for folder name
    safe_folder_name = sanitize_folder_name(agency_name)
    agency_dir = os.path.join(base_dir, safe_folder_name)
    
    # Create agency-specific directory if it doesn't exist
    if not os.path.exists(agency_dir):
        os.makedirs(agency_dir)
    
    # Create filename with timestamp and query suffix if provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{query_suffix}" if query_suffix else ""
    filename = f"{search_type}{suffix}_{timestamp}.json"
    full_path = os.path.join(agency_dir, filename)
    
    # Write response to file
    with open(full_path, 'w') as f:
        json.dump(response_data, f, indent=4)
    
    return full_path

def process_agency(agency_name):
    """
    Process a single agency: 
    1. Search for "AGENCY_NAME website"
    2. Search for "AGENCY_NAME address"
    3. Run maps search for AGENCY_NAME
    """
    results = {
        "agency_name": agency_name,
        "website_file": None,
        "address_file": None,
        "maps_file": None
    }
    
    try:
        # 1. Search for agency + "website"
        website_query = f"{agency_name} address contact site"
        # For website/contact page
        website_results = search_serper(website_query, API_KEY, "search")
        if website_results:
            best_link = extract_preferred_link(website_results)
            if best_link:
                print(f"Best link for {agency_name}: {best_link}")
            results["website_file"] = save_response(website_results, agency_name, "search", "website")
        # Add delay between search types for the same agency
        time.sleep(random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY))
        
        # 2. Search for agency + "address"
        address_query = f"{agency_name} site contact us location"
        address_results = search_serper(address_query, API_KEY, "search")
        if address_results:
            results["address_file"] = save_response(address_results, agency_name, "search", "address")
        
        # Add delay between search types for the same agency
        time.sleep(random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY))
        
        # 3. Run maps API for just the agency name
        maps_results = search_serper(agency_name, API_KEY, "maps")
        if maps_results:
            results["maps_file"] = save_response(maps_results, agency_name, "maps")
    
    except Exception as e:
        print(f"Error processing {agency_name}: {e}")
    
    return results

def extract_preferred_link(results_json):
    """
    Extract the most relevant URL based on keywords in the link.
    """
    if not results_json or "organic" not in results_json:
        return None

    # Prioritize links with these keywords
    preferred_keywords = ["contact", "location", "about", "visit", "directions"]
    
    for item in results_json["organic"]:
        link = item.get("link", "").lower()
        if any(kw in link for kw in preferred_keywords):
            return item["link"]

    # Fallback to first result
    return results_json["organic"][0]["link"] if results_json["organic"] else None

def main():
    """
    Read agencies from CSV and process each one concurrently in batches
    """
    try:
        # Path to the CSV file
        csv_file = "/Users/shreya/Documents/UCDavis/Practicum/Spring_Quarter_Practicum/Web_Scraping_MyCodes/shreyaV2/Peilu_List_Agencies/Peilu_100_MostRecentAgencies.csv"
        
        # Read agencies from CSV
        agency_names = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            # Skip header if it exists
            header = next(csv_reader, None)
            
            # Collect agency names
            for row in csv_reader:
                if row and len(row) > 0 and row[0].strip():
                    agency_names.append(row[0].strip())
        
        # Find out if there's a previous run status file to resume from
        resume_file = "resume_progressV2.json"
        processed_agencies = set()
        
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                try:
                    resume_data = json.load(f)
                    processed_agencies = set(resume_data.get("processed_agencies", []))
                    print(f"Resuming from previous run - {len(processed_agencies)} agencies already processed")
                except:
                    print("Could not load resume file, starting from beginning")
        
        # Filter out already processed agencies
        agencies_to_process = [a for a in agency_names if a not in processed_agencies]
        total_agencies = len(agencies_to_process)
        
        print(f"Found {total_agencies} agencies to process")
        
        # Process agencies in batches to better manage resources
        processed_count = 0
        
        for batch_start in range(0, total_agencies, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_agencies)
            current_batch = agencies_to_process[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(total_agencies+BATCH_SIZE-1)//BATCH_SIZE} ({len(current_batch)} agencies)")
            
            # Process the current batch concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Create a mapping of futures to agency names for better reporting
                future_to_agency = {executor.submit(process_agency, agency): agency for agency in current_batch}
                
                # Process results as they complete with a progress bar
                with tqdm.tqdm(total=len(current_batch), desc="Processing batch") as progress_bar:
                    for future in concurrent.futures.as_completed(future_to_agency):
                        agency = future_to_agency[future]
                        try:
                            result = future.result()
                            processed_agencies.add(agency)
                            processed_count += 1
                            progress_bar.update(1)
                            progress_bar.set_description(f"Processed: {agency}")
                        except Exception as e:
                            print(f"Error processing {agency}: {e}")
                            progress_bar.update(1)
                
                # Save progress after each batch
                with open(resume_file, 'w') as f:
                    json.dump({"processed_agencies": list(processed_agencies)}, f)
            
            # Pause between batches to let resources reset if there are more batches to process
            if batch_end < total_agencies:
                print(f"Pausing for {BATCH_PAUSE} seconds to release resources...")
                time.sleep(BATCH_PAUSE)
        
        print(f"All agencies have been processed! Total: {processed_count}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
