import pandas as pd

# Load the original CSV file with fish data
df = pd.read_csv('pso_fish_positions.csv')

# Create a list to store reshaped data
reshaped_data = []

# Iterate over all the iterations (rows in the CSV)
for index, row in df.iterrows():
    iteration = row['Iteration']
    
    # Iterate over each fish (1 to 50)
    for i in range(1, 51):
        fish_data = {
            'Iteration': iteration,
            'Fish_ID': i,
            'Position X': row[f'Position X{i}'],
            'Position Y': row[f'Position Y{i}'],
            'Velocity X': row[f'Velocity X{i}'],
            'Velocity Y': row[f'Velocity Y{i}'],
            'Best Position X': row[f'Best Position X{i}'],
            'Best Position Y': row[f'Best Position Y{i}'],
            'Fitness Value': row[f'Fitness Value{i}']
        }
        reshaped_data.append(fish_data)

# Convert reshaped data to a DataFrame
reshaped_df = pd.DataFrame(reshaped_data)

# Save the reshaped DataFrame to a new CSV
reshaped_df.to_csv('reshaped_fish_data.csv', index=False)

# Check the reshaped data
print(reshaped_df.head())
