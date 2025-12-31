#!/usr/bin/env python3
import csv
import re
import os
import sys
import json
import requests
import asyncio
import aiohttp
from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor
import time
from functools import partial
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Credentials from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")

# Concurrency settings from .env
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 100))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", 600))
REQUEST_INTERVAL = 60 / RATE_LIMIT_RPM  # Time between requests in seconds

# Semaphore to control concurrency
semaphore = None

def analyze_with_openai(address_from_maps, address_from_search_snippet, address_from_search_answerbox):
    """
    Use OpenAI to analyze addresses and determine confidence.
    
    Args:
        address_from_maps (str): Address from maps
        address_from_search_snippet (str): Address from search snippet
        address_from_search_answerbox (str): Address from search answerbox
        
    Returns:
        dict: A dictionary with extracted address components and confidence
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"  # OpenAI uses Bearer auth instead of api-key
    }
    
    # Prepare prompt for OpenAI
    prompt = f"""
    Analyze these potential addresses for the same location:
    
    ADDRESS FROM MAPS: "{address_from_maps}"
    ADDRESS FROM SEARCH SNIPPET: "{address_from_search_snippet}"
    ADDRESS FROM SEARCH ANSWERBOX: "{address_from_search_answerbox}"
    
    Extract the address components from the most reliable source (or a combination if needed).
    If an address component isn't found, leave it empty.
    
    Determine confidence as:
    - "high": Multiple sources contain matching address information or one source has a complete, well-formatted address
    - "medium": Only one source has address information but it appears incomplete
    - "low": The sources conflict, none contains a proper address, or the information is unclear

    Return ONLY a JSON object with these fields:
    - street_address: Street address including number and name
    - city: City name
    - state: Two-letter state code (e.g., CA, NY)
    - zip_code: Postal code
    - confidence: "high", "medium", or "low"
    """
    
    payload = {
        "model": OPENAI_MODEL,  # OpenAI requires the model in the payload
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts address components."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            OPENAI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            
            # Extract the JSON from the response
            # Sometimes OpenAI wraps the response in markdown code blocks or additional text
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
            
            # Clean up the string for JSON parsing
            json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)  # Remove non-ASCII characters
            
            # Try to parse as JSON
            try:
                results = json.loads(json_str)
                return results
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from: {json_str}")
                return get_fallback_response(address_from_maps)
        else:
            print(f"Error from OpenAI API: {response.status_code}, {response.text}")
            return get_fallback_response(address_from_maps)
    except Exception as e:
        print(f"Exception when calling OpenAI: {str(e)}")
        return get_fallback_response(address_from_maps)

async def analyze_with_openai_async(address_from_maps, address_from_search_snippet, address_from_search_answerbox):
    """
    Async version of analyze_with_openai using OpenAI to analyze addresses.
    
    Args:
        address_from_maps (str): Address from maps
        address_from_search_snippet (str): Address from search snippet
        address_from_search_answerbox (str): Address from search answerbox
        
    Returns:
        dict: A dictionary with extracted address components and confidence
    """
    async with semaphore:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"  # OpenAI uses Bearer auth instead of api-key
        }
        
        # Prepare prompt for OpenAI
        prompt = f"""
        Analyze these potential addresses for the same location:
        
        ADDRESS FROM MAPS: "{address_from_maps}"
        ADDRESS FROM SEARCH SNIPPET: "{address_from_search_snippet}"
        ADDRESS FROM SEARCH ANSWERBOX: "{address_from_search_answerbox}"
        
        Extract the address components from the most reliable source (or a combination if needed).
        If an address component isn't found, leave it empty.
        
        Determine confidence as:
        - "high": Multiple sources contain matching address information or one source has a complete, well-formatted address
        - "medium": Only one source has address information but it appears incomplete
        - "low": The sources conflict, none contains a proper address, or the information is unclear

        Return ONLY a JSON object with these fields:
        - street_address: Street address including number and name
        - city: City name
        - state: Two-letter state code (e.g., CA, NY)
        - zip_code: Postal code
        - confidence: "high", "medium", or "low"
        """
        
        payload = {
            "model": OPENAI_MODEL,  # OpenAI requires the model in the payload
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts address components."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.1
        }
        
        try:
            # Implement rate limiting with slight jitter to avoid thundering herd problem
            await asyncio.sleep(REQUEST_INTERVAL * (0.8 + 0.4 * random.random()))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OPENAI_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=15
                ) as response:
                    if response.status == 200:
                        content = await response.json()
                        content = content["choices"][0]["message"]["content"]
                        
                        # Extract the JSON from the response
                        # Sometimes OpenAI wraps the response in markdown code blocks or additional text
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = content
                        
                        # Clean up the string for JSON parsing
                        json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)  # Remove non-ASCII characters
                        
                        # Try to parse as JSON
                        try:
                            results = json.loads(json_str)
                            return results
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON from: {json_str}")
                            return await get_fallback_response_async(address_from_maps)
                    else:
                        print(f"Error from OpenAI API: {response.status}, {await response.text()}")
                        return await get_fallback_response_async(address_from_maps)
        except Exception as e:
            print(f"Exception when calling OpenAI: {str(e)}")
            return await get_fallback_response_async(address_from_maps)

# Keep the synchronous version for backward compatibility
def analyze_with_openai(address_from_maps, address_from_search_snippet, address_from_search_answerbox):
    """
    Use OpenAI to analyze addresses and determine confidence.
    Synchronous wrapper around the async implementation.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            analyze_with_openai_async(address_from_maps, address_from_search_snippet, address_from_search_answerbox)
        )
    finally:
        loop.close()

def get_fallback_response(address):
    """Fallback to basic extraction if OpenAI fails"""
    street_address, city, state, zip_code, confidence = extract_address_components(address)
    return {
        "street_address": street_address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "confidence": confidence
    }

async def get_fallback_response_async(address):
    """Async fallback to basic extraction if OpenAI fails"""
    street_address, city, state, zip_code, confidence = extract_address_components(address)
    return {
        "street_address": street_address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "confidence": confidence
    }

def extract_address_components(address):
    """
    Basic address component extraction (fallback method)
    
    Args:
        address (str): The formatted address string
        
    Returns:
        tuple: (street_address, city, state, zip_code, confidence)
    """
    confidence = "medium"  # Default to medium for the fallback method
    
    # Handle empty addresses
    if not address or address.strip() == "":
        return "", "", "", "", "low"
    
    # Common format: street, city, state zip
    parts = address.strip().split(',')
    
    # Check if we have enough parts
    if len(parts) < 2:
        return address, "", "", "", "low"
    
    # Extract street address (first part)
    street_address = parts[0].strip()
    
    # Last part typically contains state and zip
    last_part = parts[-1].strip()
    
    # Extract state and zip
    # Pattern for "STATE ZIP" or "ST ZIP"
    state_zip_pattern = r'([A-Z]{2})\s+(\d{5}(?:-\d{4})?)'
    state_zip_match = re.search(state_zip_pattern, last_part)
    
    if state_zip_match:
        state = state_zip_match.group(1)
        zip_code = state_zip_match.group(2)
    else:
        # Try to extract just the state
        state_match = re.search(r'\b([A-Z]{2})\b', last_part)
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', last_part)
        
        state = state_match.group(1) if state_match else ""
        zip_code = zip_match.group(1) if zip_match else ""
        
        if not state or not zip_code:
            confidence = "low"
    
    # City is usually the second-to-last part or part of the last part before the state
    if len(parts) >= 3:
        city = parts[-2].strip()
    else:
        # Try to extract city from the last part
        city_part = last_part.replace(state, "").replace(zip_code, "").strip()
        if city_part:
            city = city_part.strip(', ')
        else:
            city = ""
            confidence = "low"
    
    return street_address, city, state, zip_code, confidence

async def process_row_async(row):
    """Process a single row of agency data asynchronously"""
    address_from_maps = row.get('address_from_maps', '')
    address_from_search_snippet = row.get('address_from_search_snippet', '')
    address_from_search_answerbox = row.get('address_from_search_answerbox', '')
    
    # Analyze addresses with OpenAI
    results = await analyze_with_openai_async(
        address_from_maps, 
        address_from_search_snippet, 
        address_from_search_answerbox
    )
    
    # Add the new fields to the row
    row['street_address'] = results.get('street_address', '')
    row['city'] = results.get('city', '')
    row['state'] = results.get('state', '')
    row['zip_code'] = results.get('zip_code', '')
    row['address_confidence'] = results.get('confidence', 'low')
    
    print(f"Processed: {row['agency_name']}")
    return row

async def process_agency_data_async(input_file, output_file):
    """
    Process agency data CSV file to extract address components using OpenAI with concurrency.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames) + ['street_address', 'city', 'state', 'zip_code', 'address_confidence']
        rows = list(reader)
    
    print(f"Processing {len(rows)} rows with max concurrency of {MAX_CONCURRENT_REQUESTS}")
    
    # Process all rows concurrently
    tasks = [process_row_async(row) for row in rows]
    processed_rows = await asyncio.gather(*tasks)
    
    # Write results to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
    
    print(f"Processing complete. Output written to {output_file}")

def process_agency_data(input_file, output_file):
    """
    Process agency data CSV file to extract address components using OpenAI.
    This is a synchronous wrapper around the async implementation.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_agency_data_async(input_file, output_file))
    finally:
        loop.close()

def test_openai_connection():
    """
    Test the OpenAI API connection with a simple request.
    This helps verify if the API key is valid and has sufficient quota.
    """
    print("Testing OpenAI API connection...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Use a minimal prompt to minimize token usage
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return the following text as-is: 'OpenAI API connection successful'"}
        ],
        "max_tokens": 20,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            OPENAI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Success! Response: {content}")
            return True
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def main():
    # Add test mode option
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_result = test_openai_connection()
        return 0 if test_result else 1
        
    # Original function continues below
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "agency_data_preprocessed.csv"
        output_file = f"agency_data_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return 1
    
    process_agency_data(input_file, output_file)
    return 0

if __name__ == "__main__":
    sys.exit(main())
