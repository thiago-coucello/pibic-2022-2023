import os
import pandas as pd
import numpy as np

DRIVE = os.path.splitdrive(os.getcwd())[0]

RESULTS_PATH = f"{DRIVE}\\PIBIC\\2022-2023\\Results"
OUTPUT_NAME = "results"
SUBSETS = ["05", "10", "15", "20", "25", "30", "35", "40", "45",
           "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100"]

# Columns for analysis
COLUMNS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "auc",
    "val_accuracy",
    "val_precision",
    "val_sensitivity",
    "val_specificity",
    "val_f1_score",
    "val_auc"
]

subsets = os.listdir(RESULTS_PATH)
models = set()
output = open("C:\TEMP\duplicated_csvs.log", "w")
found = False

for subset in subsets:  # 05 - 100
    if subset.endswith(".csv") or subset.endswith(".txt") or subset.lower() == "metrics":
        continue

    SUBSET_RESULTS = os.path.join(RESULTS_PATH, subset)
    strategies = os.listdir(SUBSET_RESULTS)
    for strategy in strategies:  # DL - ML
        STRATEGY_CSVS = os.path.join(SUBSET_RESULTS, strategy, "csvs")
        strategy_files = os.listdir(STRATEGY_CSVS)
        results_csvs = [
            file for file in strategy_files if file.endswith(".csv")]

        for csv in results_csvs:
            model = csv.split("_")[0]
            models.add(model)
            csv_path = os.path.join(STRATEGY_CSVS, csv)

            df = pd.read_csv(csv_path)

            if df["folder"].duplicated().any():
                found = True
                output.write(csv_path + "\n")

output.close()

if not found:
    print("Nenhuma duplicata encontrada!")
else:
    print("Arquivos com duplicatas listados em C:\TEMP\duplicated_csvs.log!")
