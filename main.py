import pandas as pd

# Load the CSV file
df = pd.read_csv("C:\\Users\\ABDUR RAHMAN\\OneDrive\\Desktop\\Buddy\\Avinya25\\recipe\\back\\epi_r.csv")

# Drop rows with missing values
df_cleaned = df.dropna()

# Keep only the first 6 columns
df_cleaned = df_cleaned.iloc[:,0:6]

# Reset index and add an 'Index' column
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.insert(0, "Index", df_cleaned.index)

# Save the cleaned dataset
df_cleaned.to_csv("cleaned_file.csv", index=False)

print("Cleaned dataset with 7 columns saved as 'cleaned_file.csv'")
