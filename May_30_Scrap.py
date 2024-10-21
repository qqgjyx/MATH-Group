#!/usr/bin/env python
# coding: utf-8

# In[11]:


import requests

url = "https://mediadive.dsmz.de/strains"
response = requests.get(url)

if response.status_code == 200:
    print("Successfully fetched the webpage")
else:
    print("Failed to fetch the webpage")


# In[12]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())  # This will print the HTML structure of the page


# In[13]:


# Find the table or relevant data elements
data = []
table_rows = soup.find_all('tr')[1:]  # Skip the header row

# Loop through rows and extract data
for row in table_rows:
    columns = row.find_all('td')
    row_data = [col.get_text(strip=True) for col in columns]
    data.append(row_data)

print(data)  # Print to check the extracted data


# In[14]:


import pandas as pd

# Create a DataFrame from the extracted data
columns = ["Organism Group", "Name", "Taxonomy", "Growth Media", "External Links"]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('dsmz_strains.csv', index=False)

print("Data scraped and saved to dsmz_strains.csv")


# In[19]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to extract data from a single page
def extract_data_from_page(url, page):
    params = {"p": page}
    response = requests.get(url, params=params)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    data = []
    table_rows = soup.find_all('tr')[1:]  # Skip the header row
    for row in table_rows:
        columns = row.find_all('td')
        row_data = [col.get_text(strip=True) for col in columns]
        data.append(row_data)
    return data, soup

# Base URL of the webpage
base_url = "https://mediadive.dsmz.de/strains"
all_data = []

# Initial page
page = 1
while True:
    # Extract data from the current page
    page_data, soup = extract_data_from_page(base_url, page)
    all_data.extend(page_data)
    
    # Print the current page number
    print(f"Processing page {page}")
    
    # Find the next page button
    next_button = soup.find('button', {'title': 'next'})
    if next_button and 'disabled' not in next_button.attrs:
        page += 1
    else:
        break

# Create a DataFrame from the extracted data
columns = ["Organism Group", "Name", "Taxonomy", "Growth Media", "External Links"]
df = pd.DataFrame(all_data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('dsmz_strains.csv', index=False)

print("Data scraped and saved to dsmz_strains.csv")


# In[ ]:




