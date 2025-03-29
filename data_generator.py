import pandas as pd
import os
from faker import Faker
import random
import csv

# Setup
fake = Faker()
output_dir = "TestData/fakeData/pandasfaker"
os.makedirs(output_dir, exist_ok=True)

# Define column generators
def generate_row():
    return {
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 85),
        "address": fake.address().replace("\n", ", "),
        "phone": fake.phone_number(),
        "company": fake.company(),
        "job": fake.job(),
        "date_joined": fake.date_between(start_date='-10y', end_date='today'),
        "salary": round(random.uniform(30000, 150000), 2),
        "country": fake.country()
    }

# Generate files
for i in range(10):
    data = [generate_row() for _ in range(5000)]
    df = pd.DataFrame(data)
    filename = f"realistic_data_{i+1}.csv"
    df.to_csv(os.path.join(output_dir, filename), index=False, quoting=csv.QUOTE_ALL)
    print(f"Created: {filename}")

print("All files generated in folder:", output_dir)