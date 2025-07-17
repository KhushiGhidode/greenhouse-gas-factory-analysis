import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\lab-12\Downloads\SupplyChainGHGEmissionFactors_v1.2_NAICS_CO2e_USD2021.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Data Cleaning: Drop rows with missing values (if any)
data_cleaned = data.dropna()

# Exploratory Data Analysis (EDA)
# Summary statistics
print(data_cleaned.describe())

# Visualize the distribution of GHG emissions
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['EmissionFactor'], bins=30, kde=True)
plt.title('Distribution of GHG Emission Factors')
plt.xlabel('Emission Factor (CO2e)')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Grouping by a categorical variable (e.g., NAICS code) and calculating mean emissions
mean_emissions = data_cleaned.groupby('NAICS')['EmissionFactor'].mean().reset_index()

# Visualize mean emissions by NAICS code
plt.figure(figsize=(12, 6))
sns.barplot(x='NAICS', y='EmissionFactor', data=mean_emissions)
plt.title('Mean GHG Emission Factors by NAICS Code')
plt.xticks(rotation=90)
plt.xlabel('NAICS Code')
plt.ylabel('Mean Emission Factor (CO2e)')
plt.show()
