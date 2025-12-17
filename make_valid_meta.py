
import pandas as pd

df = pd.read_csv("OpenScreenSoundLibrary-v1/train_meta.csv")

split = int(0.9 * len(df))

train_df = df.iloc[:split]
valid_df = df.iloc[split:]

train_df.to_csv("OpenScreenSoundLibrary-v1/train_meta.csv", index=False)
valid_df.to_csv("OpenScreenSoundLibrary-v1/valid_meta.csv", index=False)

print("train_meta e valid_meta gerados corretamente")
