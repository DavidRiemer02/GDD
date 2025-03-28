from datallm import DataLLM

from mostlyai.sdk import MostlyAI

# initialize client
mostly = MostlyAI(
    api_key='mostly-2b41be09d0c53214ee7cafbec4683babacd7b4eb32bc2e3307a43b20b18af10a', 
    base_url='https://app.mostly.ai'
)

# train a generator
g = mostly.train(
    data='https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz'
)

# probe for some samples
mostly.probe(g, size=10)

# generate a synthetic dataset
sd = mostly.generate(g, size=2_000)

# start using it
sd.data()
datallm = DataLLM(api_key='mostly-2b41be09d0c53214ee7cafbec4683babacd7b4eb32bc2e3307a43b20b18af10a')

# Define the columns and their properties for the mock data generation
columns = {
    "Name": {"prompt": "A realistic first and last name", "dtype": "string"},
    "Age": {"prompt": "An age between 18 and 90", "dtype": "integer"},
    "Gender": {"prompt": "Gender of the person", "dtype": "category", "categories": ["Male", "Female", "Other"]},
    "Email": {"prompt": "A realistic email address", "dtype": "string"},
    "Phone Number": {"prompt": "A realistic phone number", "dtype": "string"},
    "Address": {"prompt": "A realistic street address", "dtype": "string"},
    "City": {"prompt": "A city name", "dtype": "string"},
    "State": {"prompt": "A state name", "dtype": "string"},
    "Country": {"prompt": "A country name", "dtype": "string"},
    "Zip Code": {"prompt": "A postal code", "dtype": "string"},
    "Date of Birth": {"prompt": "A date of birth", "dtype": "date"},
    "Occupation": {"prompt": "A job title", "dtype": "string"},
    "Company": {"prompt": "A company name", "dtype": "string"},
    "Salary": {"prompt": "An annual salary", "dtype": "float"},
    "Marital Status": {"prompt": "Marital status", "dtype": "category", "categories": ["Single", "Married", "Divorced"]}
}

# Generate the mock data
data = datallm.mock(n=20000, data_description="Realistic demographic and employment data", columns=columns, progress_bar=True)

# Save the generated data to a CSV file
data.to_csv("results/mock_data.csv", index=False)