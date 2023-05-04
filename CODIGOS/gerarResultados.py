import os
import pandas as pd
import numpy as np

DRIVE = os.path.splitdrive(os.getcwd())[0]

RESULTS_PATH = f"{DRIVE}\\PIBIC\\2022-2023\\Results"
OUTPUT_NAME = "results"
OUTPUT_EXTENSION = "csv"

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

for subset in subsets: # 05 - 100
    if subset.endswith(".csv"):
        continue

    SUBSET_RESULTS = os.path.join(RESULTS_PATH, subset)
    strategies = os.listdir(SUBSET_RESULTS)
    for strategy in strategies: # DL - ML
        STRATEGY_CSVS = os.path.join(SUBSET_RESULTS, strategy, "csvs")
        results_csvs = os.listdir(STRATEGY_CSVS)
        results_csvs = [csv for csv in results_csvs if csv.endswith(".csv")]
        for csv in results_csvs:
            model = csv.split("_")[0]
            csv_path = os.path.join(STRATEGY_CSVS, csv)
            output_file = os.path.join(RESULTS_PATH, f"{OUTPUT_NAME}_{subset}.{OUTPUT_EXTENSION}")
            
            df = pd.read_csv(csv_path)

            if df["folder"].size < 100:
                # Não processou direito o subconjunto
                continue

            result_df = pd.DataFrame()
            runtime = df["runtime"].cumsum().iloc[-1]
            val_runtime = df["val_runtime"].cumsum().iloc[-1]
            total_runtime = runtime + val_runtime
            result_data = {"subset": subset, "strategy": strategy, "model": model, "runtime": runtime, "val_runtime": val_runtime, "total_runtime": total_runtime}
            for column in COLUMNS:
                first_quartile, third_quartile = df[column].quantile([0.25, 0.75])
                iqr = third_quartile - first_quartile
                column_median = df[column].median()
                lower_whisker = 100 * (first_quartile - 1.5 * iqr)
                upper_whisker = 100 * (third_quartile + 1.5 * iqr)

                result_data[f"{column}_median"] = 100 * column_median
                result_data[f"{column}_lower"] = 100 * first_quartile
                result_data[f"{column}_upper"] = 100 * third_quartile
                result_data[f"{column}_lower_whisker"] = 0 if lower_whisker < 0 else lower_whisker
                result_data[f"{column}_upper_whisker"] = 100 if upper_whisker > 100 else upper_whisker
                
                # Formatando para gráfico de linhas
                coordinates = ""
                for (epoch, value) in zip(df["folder"], df[column]):
                    coordinates += f"({epoch}, {100 * value})"
                result_data[f"{column}_coords"] = coordinates

            result_temp_df = pd.DataFrame([result_data])
            result_df = pd.concat([result_df, result_temp_df], ignore_index = True)
            #result_df: pd.DataFrame = result_df.append(result_data, ignore_index = True)
            result_df.to_csv(output_file, index=None)
            
