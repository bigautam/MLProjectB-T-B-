import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score

df = pd.read_csv("/mnt/c/Users/Bethl/MLProjectB-T-B-/PS_20174392719_1491204439457_log.csv")
# print(df.head(5))

#cleaning up the step so that they are in a 24hr interval instead of up to 744
df["step"] = df["step"] % 24
# Convert any 0s (which come from multiples of 24) to 24
df.loc[df["step"] == 0, "step"] = 24

#checking in the og this step was 13 now its 9 so it works :)
# print(df.iloc[591705])     

#now we only care for merchants so in the nameDest we want to drop the rows that start with C 
# since thats costumers

# Create a boolean mask for rows where the first letter is 'C'
first_letter_is_C = df['nameDest'].str[0] == 'C'

# Invert the mask to get rows where the first letter is NOT 'C'
not_C_first_letter = ~first_letter_is_C

# Subset the DataFrame using the inverted mask
df_dropped = df[not_C_first_letter]

# Reset the index
df_dropped = df_dropped.reset_index(drop=True)

print(df_dropped["nameDest"])