import pandas as pd
import os

DATAFOLDER = "datatest"

OUTPUT_FILE = "subjects.csv"

all = []
for subfolder in os.listdir(DATAFOLDER):
    files = os.path.join(DATAFOLDER, subfolder)
    for subject in os.listdir(files):
        file = os.path.join(DATAFOLDER, subfolder, subject)
        data = pd.read_csv(file)
        data["subject"] = subject
        all.append(data)

result = pd.concat(all)
result.to_csv(os.path.join(DATAFOLDER, OUTPUT_FILE), index=False)

print("done")
