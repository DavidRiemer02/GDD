import pandas as pd
import numpy as np
import random

# Sample data for cities
cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
    "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus",
    "Charlotte", "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington"
]

states = [
    "NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA", "TX", "FL", "TX", "OH", "NC",
    "CA", "IN", "WA", "CO", "DC"
]

# Generate synthetic dataset
n = 1000
data = {
    "City": [random.choice(cities) for _ in range(n)],
    "State": [states[cities.index(city)] for city in np.random.choice(cities, n)],
    "Population": np.random.randint(20000, 1000000, size=n),
    "Area_sq_km": np.round(np.random.uniform(30, 800, size=n), 2),
    "Density_per_km2": lambda df: np.round(df["Population"] / df["Area_sq_km"], 2),
    "Latitude": np.round(np.random.uniform(25.0, 49.0, size=n), 5),
    "Longitude": np.round(np.random.uniform(-125.0, -67.0, size=n), 5),
    "Elevation_m": np.random.randint(0, 2000, size=n),
    "Is_Capital": [random.choice([True, False]) for _ in range(n)],
    "Average_Income_USD": np.random.randint(30000, 150000, size=n)
}

# Convert to DataFrame
df = pd.DataFrame(data)
df["Density_per_km2"] = np.round(df["Population"] / df["Area_sq_km"], 2)

# Save to CSV
output_path = "TestData/fakeData/synthetic_city_data3.csv"
df.to_csv(output_path, index=False)

print(f"CSV with {n} rows and 10 columns saved to {output_path}")
