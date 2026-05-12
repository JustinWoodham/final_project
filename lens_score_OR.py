import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import numpy as np
import pandas as pd
from lens import download_model, LENS

# Read the CSV
csv_path = sys.argv[1] if len(sys.argv) > 1 else "lm_output.csv"
df = pd.read_csv(csv_path)

# match the column names from pipeline CSV output
sources = df["Original"].tolist()
predictions = df["LM Output"].tolist()
#nested list
references = [[s] for s in df["Reference"].tolist()]

#Load LENS and score
lens_path = download_model("davidheineman/lens")
lens_model = LENS(lens_path, rescale=True)
# run using CPU, saves package issue with CUDA
scores = lens_model.score(sources, predictions, references, devices =[0])

# used to gather few-shot quality metrics
if "--per-row" in sys.argv:
    print("\n".join(map(str, scores)))
else:
    # returns average for metric score
    print(np.mean(scores))

# if for any reason lens fails, possibly due to a cache wipe, run command, below with a test csv, and then set devices = None or 0 or 1 or remove
# python lens_score.py csv/lens_test.csv
