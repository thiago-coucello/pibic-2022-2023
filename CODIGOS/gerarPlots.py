import os
import pandas as pd
import numpy as np

DRIVE = os.path.splitdrive(os.getcwd())[0]

RESULTS_PATH = f"{DRIVE}\\PIBIC\\2022-2023\\Results"

# DONE: 'DenseNet201',
# 'MobileNet' (1-50), 'ResNet101V2' (1-50), "ResNet50V2" (1-50), "ResNet152V2" (1-50),
# "MobileNetV2" (1-40), "VGG16" (1-40), "VGG19" (1-40), "ResNet50" (1-40),
# "ResNet101" (1-50), "ResNet152" (1-50), "Xception" (1-50), "EfficientNetB4" (1-50),
# "InceptionV3" (1-40)

MODELS = [
    "DenseNet201",
    "MobileNet",
    "ResNet101V2",
    "ResNet50V2",
    "ResNet152V2",
    "MobileNetV2",
    "VGG16",
    "VGG19",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "Xception",
    "EfficientNetB4",
    "InceptionV3"
]


""" 
MODELS = [
    "DenseNet201",
    "MobileNet",
    "ResNet152V2",
    "Xception",
]
 """

""" 
COLUMNS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "auc",
    "npv",
    "mcc",
    "val_accuracy",
    "val_precision",
    "val_sensitivity",
    "val_specificity",
    "val_f1_score",
    "val_auc",
    "val_npv",
    "val_mcc"
]
 """

COLUMNS = [
    "val_accuracy_mean",
    "val_precision_mean",
    "val_sensitivity_mean",
    "val_specificity_mean",
    "val_f1_score_mean",
    "val_auc_mean",
    "runtime",
    "val_runtime",
    "total_runtime"
]

ORDER_BY_COLUMNS = [c.replace("_mean", "")
                    for c in COLUMNS if not c.endswith("runtime")]

df = pd.DataFrame(columns=ORDER_BY_COLUMNS)

# TRECHO PARA GERAR PLOTS POR MODELOS
for model in MODELS:
    # Monta o nome do arquivo de resultados do modelo (model) e subconjunto (sub)
    model_metrics = os.path.join(RESULTS_PATH, f"results_{model}.csv")
    val_dict = {}

    # Criando pasta de métricas para organização
    if not os.path.isdir(os.path.join(RESULTS_PATH, "Metrics")):
        os.mkdir(os.path.join(RESULTS_PATH, "Metrics"))

    metrics_file = os.path.join(
        RESULTS_PATH, "Metrics", f"{model}_metrics.txt")

    if os.path.isfile(metrics_file):
        output = open(metrics_file, "a")
    else:
        output = open(metrics_file, "w")

    for column in COLUMNS:
        coordinates = ""

        if os.path.exists(model_metrics):
            model_metrics_df = pd.read_csv(model_metrics)
            metric_mean = model_metrics_df[column].mean()

            val_dict["model"] = model
            val_dict[column.replace("_mean", "")] = metric_mean

            for index, row in model_metrics_df.iterrows():
                subset = int(row["subset"])
                data = row[column]
                coordinates += f"{subset/100} {data}\n"

            # coordinates_output = f"%{model} - {column}\n\\addplot coordinates {'{'}\n{coordinates}{'};'}\n"
            # output.write(coordinates_output)
            output.write(f"{column}\n{coordinates}")

    temp = pd.DataFrame([val_dict])
    df = pd.concat([df, temp])
    output.close()

sorted_df = df.sort_values(
    by=ORDER_BY_COLUMNS)

sorted_df.set_index("model")

order_file = os.path.join(
    RESULTS_PATH, "Metrics", f"order.csv")

sorted_df.to_csv(order_file, index=False)
