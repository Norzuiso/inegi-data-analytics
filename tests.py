import pandas as pd
import os

# Sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Renaming columns
df.rename(columns={'A': 'X', 'B': 'Y', 'C': 'Z'}, inplace=True)
print(df)

segments_vivienda = {
                    "P1_1": [1, 2, 3, 4], 
                    "P1_2": [1, 2, 3, 4],
                    "P1_3": []
                    }
base_file_name = os.path.splitext(os.path.basename("CN_VIVIENDAS.csv"))[0]

print (base_file_name)