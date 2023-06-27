import os
import pandas as pd
import numpy as np

DRIVE = os.path.splitdrive(os.getcwd())[0]

RESULTS_PATH = f"{DRIVE}\\PIBIC\\2022-2023\\Results"
OUTPUT_NAME = "results"
SUBSETS = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100"]

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

for subset in subsets: # 05 - 100
    if subset.endswith(".csv") or subset.endswith(".txt") or subset.lower() == "metrics":
        continue

    SUBSET_RESULTS = os.path.join(RESULTS_PATH, subset)
    strategies = os.listdir(SUBSET_RESULTS)
    for strategy in strategies: # DL - ML
        STRATEGY_CSVS = os.path.join(SUBSET_RESULTS, strategy, "csvs")
        strategy_files = os.listdir(STRATEGY_CSVS)
        results_csvs = [file for file in strategy_files if file.endswith(".csv")]

        for csv in results_csvs:
            model = csv.split("_")[0]
            models.add(model)
            csv_path = os.path.join(STRATEGY_CSVS, csv)
            output_file = os.path.join(RESULTS_PATH, f"{OUTPUT_NAME}_{model}_{subset}.csv")
            
            df = pd.read_csv(csv_path)

            """ 
            if df["folder"].size < 100:
                # Não processou direito o subconjunto
                continue
            """
    
            result_df = pd.DataFrame()
            runtime = df["runtime"].cumsum().iloc[-1]
            val_runtime = df["val_runtime"].cumsum().iloc[-1]
            total_runtime = runtime + val_runtime
            result_data = {"subset": subset, "strategy": strategy, "model": model, "runtime": runtime, "val_runtime": val_runtime, "total_runtime": total_runtime}
            
            for column in COLUMNS:
                first_quartile, third_quartile = df[column].quantile([0.25, 0.75])
                iqr = third_quartile - first_quartile
                column_median = df[column].median()
                column_mean = df[column].mean()

                lower_whisker = 100 * (first_quartile - 1.5 * iqr)
                upper_whisker = 100 * (third_quartile + 1.5 * iqr)

                result_data[f"{column}_mean"] = 100 * column_mean
                result_data[f"{column}_median"] = 100 * column_median
                result_data[f"{column}_lower"] = 100 * first_quartile
                result_data[f"{column}_upper"] = 100 * third_quartile
                result_data[f"{column}_lower_whisker"] = lower_whisker
                result_data[f"{column}_upper_whisker"] = upper_whisker
                
                # Formatando para gráfico de linhas
                """ 
                coordinates = ""
                for (partition, value) in zip(df["folder"], df[column]):
                    coordinates += f"({partition}, {100 * value})\n"
                """

            result_temp_df = pd.DataFrame([result_data])
            result_df = pd.concat([result_df, result_temp_df], ignore_index = True)
            #result_df: pd.DataFrame = result_df.append(result_data, ignore_index = True)
            result_df.to_csv(output_file, index=None)


# Reiterando todos os resultados dos modelos para gerar o gráfico de linha
print(models)
for model in models:
    for column in COLUMNS:
        coordinates = ""
        for sub in SUBSETS:
            # Monta o nome do arquivo de resultados do modelo (model) e subconjunto (sub)
            model_metrics = os.path.join(RESULTS_PATH, f"{OUTPUT_NAME}_{model}_{sub}.csv")
            
            # Se o arquivo existe processa e gera gráfico de linha
            if os.path.exists(model_metrics):
                model_metrics_df = pd.read_csv(model_metrics)
                data = model_metrics_df[f"{column}_mean"][0]
                coordinates += f"\t({sub}, {data})\n"
        
        # Criando pasta de métricas para organização
        if not os.path.isdir(os.path.join(RESULTS_PATH, "Metrics")):
            os.mkdir(os.path.join(RESULTS_PATH, "Metrics"))
        
        metrics_file = os.path.join(RESULTS_PATH, "Metrics", f"{model}_{column}.txt")
        coordinates_output = f"%{model}\n\\addplot coordinates {'{'}\n{coordinates}{'};'}\n"

        with open(metrics_file, "w") as metrics_output:
            metrics_output.write(coordinates_output)  