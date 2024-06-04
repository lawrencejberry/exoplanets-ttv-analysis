import pandas as pd
import re

def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", names=["index", "epoch", "transit_time", "error", "data_quality", "observer"])
    df["transit_time"] = df["transit_time"] - df["transit_time"].quantile(0.5)
    df["epoch"] = df["epoch"] - df["epoch"].quantile(0.5)
    return df


def load_nasa_data(path):
    metadata = []
    with open(path) as f:
        header = 0
        while True:
            line = next(f)
            if line.startswith("#"):
                metadata.append(line)
                header += 1
            else:
                break
                
    column_name_map = {k: v for k, v in re.findall("COLUMN\s+(\w+):\s+([^,]+)", "".join(metadata))}
    df = pd.read_csv(path, skiprows=header, low_memory=False)
    df = df.rename(columns=column_name_map)

    return df
