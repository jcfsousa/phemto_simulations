#!/usr/bin/env python3
import sys
import os
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 to_csv.py input.parquet")
        sys.exit(1)

    in_path = sys.argv[1]

    if not os.path.isfile(in_path):
        print(f"Error: file not found: {in_path}")
        sys.exit(1)

    # Build output name: replace .parquet by .csv
    base, ext = os.path.splitext(in_path)
    if ext.lower() == ".parquet":
        out_path = base + ".csv"
    else:
        out_path = in_path + ".csv"

    print(f"Reading parquet: {in_path}")
    df = pd.read_parquet(in_path)

    print(f"Writing csv:     {out_path}")
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
