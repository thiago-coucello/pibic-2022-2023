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
    "auc",
    "val_auc"
]

train_dict = {}
val_dict = {}

# TRECHO PARA GERAR PLOTS POR MODELOS
for model in MODELS:
    # Monta o nome do arquivo de resultados do modelo (model) e subconjunto (sub)
    model_metrics = os.path.join(RESULTS_PATH, f"results_{model}.csv")

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
            auc_mean = model_metrics_df["auc_mean"].mean()
            val_auc_mean = model_metrics_df["val_auc_mean"].mean()

            train_dict[model] = auc_mean
            val_dict[model] = val_auc_mean

            for index, row in model_metrics_df.iterrows():
                subset = int(row["subset"])
                data = row[f"{column}_mean"]
                coordinates += f"{subset/100} {data}\n"

            # coordinates_output = f"%{model} - {column}\n\\addplot coordinates {'{'}\n{coordinates}{'};'}\n"
            # output.write(coordinates_output)
            output.write(f"{column}\n{coordinates}")

    output.close()

sorted_train_dict = sorted(train_dict.items(), key=lambda x: x[1])
sorted_val_dict = sorted(val_dict.items(), key=lambda x: x[1])

print("Sorted train:", sorted_train_dict)
print()
print("Sorted val:", sorted_val_dict)
