import sys

import pandas as pd

df = pd.DataFrame()

for i in range(1, len(sys.argv)):
    df0 = pd.read_csv(sys.argv[i], engine='python', header=None)
    df = pd.concat([df, df0])
df[0] = pd.RangeIndex(start=0, stop=len(df))

df.to_csv(sys.stdout, header=False, index=False)
