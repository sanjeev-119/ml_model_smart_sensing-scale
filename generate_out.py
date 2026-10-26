
import numpy as np
import pandas as pd
from features import foot_features

# Load one 32x24 thermal frame (24 rows x 32 columns)
temp = pd.read_csv("temperature_frame.csv", header=None).to_numpy(dtype=float)

if temp.shape != (24, 32):
    raise ValueError(
        f"Expected a 24x32 frame, but got {temp.shape}"
    )

# Divide the frame into four equal quadrants
rows, cols = temp.shape
mid_row = rows // 2
mid_col = cols // 2

masks = {
    "LU": np.zeros(temp.shape, dtype=bool),
    "LL": np.zeros(temp.shape, dtype=bool),
    "RU": np.zeros(temp.shape, dtype=bool),
    "RL": np.zeros(temp.shape, dtype=bool),
}

masks["LU"][:mid_row, :mid_col] = True
masks["LL"][mid_row:, :mid_col] = True
masks["RU"][:mid_row, mid_col:] = True
masks["RL"][mid_row:, mid_col:] = True

# Extract 12 features
features, sizes = foot_features(temp, masks)

# Save features to out.csv
pd.DataFrame([features]).to_csv("out.csv", index=False)

print("Features saved to out.csv")
print(pd.DataFrame([features]).to_string(index=False))
print("Zone pixel counts:", sizes)
