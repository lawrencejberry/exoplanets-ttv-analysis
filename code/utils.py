import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", names=["index", "epoch", "transit_time", "error", "data_quality", "observer"])
    df["transit_time"] = df["transit_time"] - df["transit_time"].quantile(0.5)
    df["epoch"] = df["epoch"] - df["epoch"].quantile(0.5)
    return df
