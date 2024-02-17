#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-17 11:23:39 (ywatanabe)"

import requests
from bs4 import BeautifulSoup


# Function to get the redirected URL and then the BibTeX data
def get_bibtex(dataset_id):
    base_url = f"https://openneuro.org/datasets/{dataset_id}"
    response = requests.get(base_url)

    if response.history:
        # If there is redirection, the final URL is taken
        final_url = response.url
    else:
        # No redirection, the base URL is used
        final_url = base_url

    # Now we get the page content from the final URL
    page_content = requests.get(final_url).text
    soup = BeautifulSoup(page_content, "html.parser")

    import ipdb

    ipdb.set_trace()
    # Find the BibTeX section (you'll need to inspect the actual web page and possibly adjust this part)
    # This is a placeholder; you must identify the correct HTML element containing the BibTeX
    bibtex_section = soup.find("div", class_="bibtex-section")

    # Extract the BibTeX text (again, you may need to adjust the selector based on the web page structure)
    bibtex_text = bibtex_section.get_text(strip=True)

    return bibtex_text


# List of dataset IDs
dataset_ids = [
    "ds004100",
    "another-dataset-id",
    "yet-another-dataset-id",
]  # Add your dataset IDs here

# Dictionary to hold the BibTeX entries
bibtex_entries = {}

# Loop through each dataset ID and get the BibTeX data
for dataset_id in dataset_ids:
    bibtex_entries[dataset_id] = get_bibtex(dataset_id)

# Print out the BibTeX entries
for dataset_id, bibtex in bibtex_entries.items():
    print(f"BibTeX for {dataset_id}:")
    print(bibtex)
    print("\n")
