#test4.py was written: test3.py was copied and then hitting gpt-4o-mini model code was added
import asyncio
from crawl4ai import *
import regex as re
import os
import sys
import json
import random
import aiohttp
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from typing import List

class CustomMarkdownGenerator(DefaultMarkdownGenerator):
    def __init__(self):
        super().__init__(options={
            "preserve_inline_tags": ["span", "a"],  # Preserve span tags
            "content_source" : "raw_html",
            "escape_html": False,
            "ignore_links": False,
            "body_width": 0  # No line wrapping
        })

async def main(url, no_postal_code = False, magic = True, simulate_user = True):
    md_generator = CustomMarkdownGenerator()
    browser_config = BrowserConfig(
        browser_type="chromium",  # Options: "chromium", "firefox", "webkit"
        headless=True,            # Set to False if you want to see the browser window
        viewport_width=1280,
        viewport_height=800,
        verbose=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        # user_agent="Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" Was added for 
        # https://www.mapquest.com/us/california/elk-valley-head-start-12129364
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawler_config = CrawlerRunConfig(
            screenshot=False,
            verbose=True,
            cache_mode=CacheMode.DISABLED,
            wait_until='domcontentloaded',
            delay_before_return_html = 5,
            log_console=True,
            exclude_social_media_links=True,
            # exclude_external_links = True,
            exclude_external_images  =True,
            magic=magic,
            simulate_user=simulate_user, 
            scan_full_page=True,  # Tells the crawler to try scrolling the entire page
            scroll_delay=1,     # Delay (seconds) between scroll steps
            markdown_generator=md_generator
        )
        result = None
        try:
            result = await crawler.arun(
                url=url,
                config=crawler_config
            )
        except Exception as e:
        # if not result.success:
            print(f"Crawl Exception: {e}")
        if result is None or not getattr(result, 'markdown', None):
            print(f"Crawl failed or no markdown for {url}")
            return ['URL not found']

        # print(result.markdown)
        # print(result.html)
        
        # print(result.extracted_content)
        # print(result)

        state_codes = [
        " AL ", " AK ", " AZ ", " AR ", " CA ", " CO ", " CT ", " DE ", " FL ", " GA ",
        " HI ", " ID ", " IL ", " IN ", " IA ", " KS ", " KY ", " LA ", " ME ", " MD ",
        " MA ", " MI ", " MN ", " MS ", " MO ", " MT ", " NE ", " NV ", " NH ", " NJ ",
        " NM ", " NY ", " NC ", " ND ", " OH ", " OK ", " OR ", " PA ", " RI ", " SC ",
        " SD ", " TN ", " TX ", " UT ", " VT ", " VA ", " WA ", " WV ", " WI ", " WY ",
        " al ", " ak ", " az ", " ar ", " ca ", " co ", " ct ", " de ", " fl ", " ga ",
        " hi ", " id ", " il ", " in ", " ia ", " ks ", " ky ", " la ", " me ", " md ",
        " ma ", " mi ", " mn ", " ms ", " mo ", " mt ", " ne ", " nv ", " nh ", " nj ",
        " nm ", " ny ", " nc ", " nd ", " oh ", " ok ", " or ", " pa ", " ri ", " sc ",
        " sd ", " tn ", " tx ", " ut ", " vt ", " va ", " wa ", " wv ", " wi ", " wy ",
        " AL", " AK", " AZ", " AR", " CA", " CO", " CT", " DE", " FL", " GA",
        " HI", " ID", " IL", " IN", " IA", " KS", " KY", " LA", " ME", " MD",
        " MA", " MI", " MN", " MS", " MO", " MT", " NE", " NV", " NH", " NJ",
        " NM", " NY", " NC", " ND", " OH", " OK", " OR", " PA", " RI", " SC",
        " SD", " TN", " TX", " UT", " VT", " VA", " WA", " WV", " WI", " WY",
        " al", " ak", " az", " ar", " ca", " co", " ct", " de", " fl", " ga",
        " hi", " id", " il", " in", " ia", " ks", " ky", " la", " me", " md",
        " ma", " mi", " mn", " ms", " mo", " mt", " ne", " nv", " nh", " nj",
        " nm", " ny", " nc", " nd", " oh", " ok", " or", " pa", " ri", " sc",
        " sd", " tn", " tx", " ut", " vt", " va", " wa", " wv", " wi", " wy",
        "AL ", "AK ", "AZ ", "AR ", "CA ", "CO ", "CT ", "DE ", "FL ", "GA ",
        "HI ", "ID ", "IL ", "IN ", "IA ", "KS ", "KY ", "LA ", "ME ", "MD ",
        "MA ", "MI ", "MN ", "MS ", "MO ", "MT ", "NE ", "NV ", "NH ", "NJ ",
        "NM ", "NY ", "NC ", "ND ", "OH ", "OK ", "OR ", "PA ", "RI ", "SC ",
        "SD ", "TN ", "TX ", "UT ", "VT ", "VA ", "WA ", "WV ", "WI ", "WY ",
        "al ", "ak ", "az ", "ar ", "ca ", "co ", "ct ", "de ", "fl ", "ga ",
        "hi ", "id ", "il ", "in ", "ia ", "ks ", "ky ", "la ", "me ", "md ",
        "ma ", "mi ", "mn ", "ms ", "mo ", "mt ", "ne ", "nv ", "nh ", "nj ",
        "nm ", "ny ", "nc ", "nd ", "oh ", "ok ", "or ", "pa ", "ri ", "sc ",
        "sd ", "tn ", "tx ", "ut ", "vt ", "va ", "wa ", "wv ", "wi ", "wy ",
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
        "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", 
        "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
        "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
        "Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", 
        "Wyoming"] 
        lines = result.markdown.split("\n")

        address_elements = ["Street", "Avenue", "Ave", "Boulevard", "Blvd", "Road", "Rd", "Drive", "Lane", "Court", "City", "Square", "Place", "Trail", "Parkway"]

        blocks_containing_addresses = []
        i = 0
        global j
        j = 0 # j is introduced here so that if addresses are on consecutive lines in markdown, for one address chunk,
            # line from previous address chunk is not taken
        found = False
        while i < len(lines):
            print("In line: " + str(i))
            # postal_code_extended_matches = re.findall(r'\b\d{5}(-\d{4})?\b', re.sub(r'[_*]', '', lines[i]))
            postal_code_extended_matches = re.findall(r'\b\d{5}(-\d{4})?', re.sub(r'[_*]', '', lines[i]))
            # Above change to regex was done after observing https://www.uclahealth.org/hospitals/santa-monica
            # ZIP CODE was followed by literal 'DIRECTIONS'. Hence, word boundary at end of ZIPCODE restriction was removed 
            if found and len(postal_code_extended_matches) > 0:
                print("I came here")
                print(f"line: {re.sub(r'[_*]', '', lines[i])}")
                if j - 1 >= max(i - 1, 0):
                    start = max(i, 0)
                elif j - 1 >= max(i - 2, 0):
                    start = max(i - 1, 0) #address could be split between lines
                else: start = max(i - 2, 0)
                postal_code_extended_matches_Next = []
                if i+2 < len(lines):
                    # postal_code_extended_matches_Next = re.findall(r'\b\d{5}(-\d{4})?\b', re.sub(r'[_*]', '', lines[i+2]))
                    postal_code_extended_matches_Next = re.findall(r'\b\d{5}(-\d{4})?', re.sub(r'[_*]', '', lines[i+2]))
                if len(postal_code_extended_matches_Next) > 0: 
                    end = min(i + 1 + 1 + 1, len(lines)) #last +1 is just because end is not included while slicing list
                    snippet = lines[start:end]
                    joined_block = "\n".join(snippet)
                    print("Start:" + str(start) + " End: " + str(end))
                    if any(state_code in joined_block for state_code in state_codes):
                        blocks_containing_addresses.append(joined_block) # There is a match in this block of text.
                    i += 3
                    j = i
                    found = True
                else:
                    end = min(i + 1 + 1, len(lines))
                    snippet = lines[start:end]
                    joined_block = "\n".join(snippet)
                    if any(state_code in joined_block for state_code in state_codes):
                        blocks_containing_addresses.append(joined_block) # There is a match in this block of text.
                    i += 2
                    j = i
                    found = True
            elif not found and len(postal_code_extended_matches) > 0: 
            # This elif block was introduced to differentiate between pages which had addresses on consecutive lines vs. not
                print("I came here")
                print(f"line: {re.sub(r'[_*]', '', lines[i])}")
                if j - 1 > max(i - 1, 0):
                    start = max(i, 0)
                elif j - 1 > max(i - 2, 0):
                    start = max(i - 1, 0) #address could be split between lines
                else: start = max(i - 2, 0)
                postal_code_extended_matches_Next = []
                if i+2 < len(lines):
                    # postal_code_extended_matches_Next = re.findall(r'\b\d{5}(-\d{4})?\b', re.sub(r'[_*]', '', lines[i+2]))
                    postal_code_extended_matches_Next = re.findall(r'\b\d{5}(-\d{4})?', re.sub(r'[_*]', '', lines[i+2]))
                if len(postal_code_extended_matches_Next) > 0: 
                    end = min(i + 1 + 1 + 1, len(lines)) #last +1 is just because end is not included while slicing list
                    snippet = lines[start:end]
                    joined_block = "\n".join(snippet)
                    print("Start:" + str(start) + " End: " + str(end))
                    if any(state_code in joined_block for state_code in state_codes):
                        blocks_containing_addresses.append(joined_block) # There is a match in this block of text.
                    i += 3
                    j = i
                    found = True
                else:
                    end = min(i + 1 + 1, len(lines))
                    snippet = lines[start:end]
                    joined_block = "\n".join(snippet)
                    print("Start:" + str(start) + " End: " + str(end))
                    if any(state_code in joined_block for state_code in state_codes):
                        blocks_containing_addresses.append(joined_block) # There is a match in this block of text.
                    i += 2
                    j = i
                    found = True
            else:
                if no_postal_code and any(address_regex in lines[i] for address_regex in address_elements): 
                    blocks_containing_addresses.append(lines[i])
                    found = True
                i += 1
                j = i
                found = False
                # print(joined_block)
                # print("\n" + "-"*40 + "\n")


            # Call OpenAI and ask it if this contains postal codes or not. If not, then return None. If it contains postal codes, return them separated by ;.
            # continue
            # OpenAI API Setup
        print(blocks_containing_addresses)
        for block in blocks_containing_addresses:
            print(block)
            print("*****************************")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_MODEL = "gpt-4o"
        OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

        MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 100))
        RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", 600))
        REQUEST_INTERVAL = 60 / RATE_LIMIT_RPM

        global semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        def return_address_prompt(block):
            return f"""
            Input block of text is: {block}

            Question:
            Does this block of text contain a valid postal address
            - If yes, return the full address as list of strings
            - If no, return None
            """
        async def process_row_async(row):
            async with semaphore:
                prompt = return_address_prompt(row)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }

                payload = {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You read a block of text and extract USA postal addresses, if any from the block of text."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format":{
                        "type": "json_schema",
                        "json_schema": {
                            "name": "addresses_response",
                            "strict": False,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "addresses": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "addressLine1": {
                                                    "type": "string",
                                                    "description": "The address line extracted from block of text."
                                                },
                                                "city": {
                                                    "type": "string",
                                                    "description": "The city of the address."
                                                },
                                                "postal_code": {
                                                    "type": "string",
                                                    "description": "Postal code (5 digit or 5+4 digits)"
                                                },
                                                "state": {
                                                    "type": "string",
                                                    "description": "State code in 2 letter code."
                                                },
                                            },
                                            "required": ["addressLine1", "postal_code", "state", "city"]
                                        }
                                    }
                                }
                                        
                            }
                        }
                    },
                    "max_tokens": 600,
                    "temperature": 0.2
                }

                await asyncio.sleep(REQUEST_INTERVAL * (0.8 + 0.4 * random.random()))

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(OPENAI_ENDPOINT, headers=headers, json=payload, timeout=20) as response:
                            if response.status == 200:
                                result = await response.json()
                                text_response = result["choices"][0]["message"]["content"]
                                # print(f"\n Block: {row}\n Extracted address: {text_response}")
                                try:
                                    json_output = json.loads(text_response)
                                except json.JSONDecodeError:
                                    print(f"Warning: Could not parse JSON, returning raw text instead:\n{text_response}")
                                    json_output = []
                                return json_output
                            else:
                                print(f"OpenAI API Error {response.status}: {await response.text()}")
                                return None
                except Exception as e:
                    print(f"Error processing row: {e}")
                    return None


        async def process_disaster_data_async():
            tasks = [process_row_async(block) for block in blocks_containing_addresses]
            processed_rows = await asyncio.gather(*tasks)
            return processed_rows

        processed_rows = await process_disaster_data_async()
        print(processed_rows)
        return processed_rows


if __name__ == "__main__":
    list_of_lists = []
    input_file = "/Users/shreya/Downloads/agency_data_preprocessedV2_Peilu_ToGiveToCharon-2.csv"
    df = pd.read_csv(input_file)
    urls = df["website"]
    for url in urls:
        processed_rows = asyncio.run(main(url, no_postal_code = False, magic = True, simulate_user = True))
        if processed_rows == []:
            processed_rows = asyncio.run(main(url, no_postal_code = False, magic = False, simulate_user = False))
        if processed_rows == []:
            processed_rows = asyncio.run(main(url, no_postal_code = True, magic = True, simulate_user = True))
        if processed_rows == []:
            processed_rows = asyncio.run(main(url, no_postal_code = True, magic = False, simulate_user = False))
        list_of_lists.append(processed_rows)
    df["address"] = list_of_lists
    df.to_csv('/Users/shreya/Downloads/Returned_address_Web_scraping.csv', index = False)
    # url = "https://www.uclahealth.org/hospitals/santa-monica"
    # processed_rows = asyncio.run(main(url, no_postal_code = False, magic = True, simulate_user = True))
    # if processed_rows == []:
    #     processed_rows = asyncio.run(main(url, no_postal_code = False, magic = False, simulate_user = False))
    # if processed_rows == []:
    #     processed_rows = asyncio.run(main(url, no_postal_code = True, magic = True, simulate_user = True))
    # if processed_rows == []:
    #     processed_rows = asyncio.run(main(url, no_postal_code = True, magic = False, simulate_user = False))



# The elif not found and len(postal_code_extended_matches) > 0: 
# block was introduced because address layout on https://www.ohsu.edu/visit/doernbecher-childrens-hospital was observed.
# The last line, processed_rows = asyncio.run(main(url, magic = False, simulate_user = False)) was
# introduced after observing these 2 webpages, as magic = True and simulate_user = True was leading to a click on webpage and a subpage wsas being opened.
# Check headless = False to notice this.
# https://www.legacyhealth.org/Doctors-and-Locations/hospitals/legacy-emanuel-medical-center
# https://www.stlukesonline.org/communities-and-locations/facilities/clinics/st-lukes-childrens-treasure-valley-pediatrics-meridian"
# https://www.legacyhealth.org/Doctors-and-Locations/hospitals/randall-childrens-hospital-at-legacy-emanuel

# The address layout on https://intermountainhealthcare.org/locations/lds-hospital was observed
# No Postal code was found inside address. Hence, no_postal_code = True line inside main was executed