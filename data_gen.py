import numpy as np
import pandas as pd
import sys

n = int(sys.argv[1])     #dataset size
dim = int(sys.argv[2])   #dataset dimension
rang = int(sys.argv[3])
fname = sys.argv[4]      #dataset name

points_very_large_no_header = np.random.rand(n, dim) * rang

# Convert to DataFrame
df = pd.DataFrame(points_very_large_no_header)

# Save to CSV without headers
df.to_csv(fname, index=False, header=False)