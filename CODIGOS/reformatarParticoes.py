import os
import pandas as pd

subsets_base_path = "C:/PIBIC/2022-2023/Datasets/"
partitions_base_path = "C:/PIBIC/2022-2023/Partitions/"
partitions = os.listdir(partitions_base_path)
subsets = os.listdir(subsets_base_path)

for subset in subsets:
    print(f"Subconjunto {subset}")
    
    if not os.path.isdir(os.path.join(subsets_base_path, subset, "Partitions")):
        os.mkdir(os.path.join(subsets_base_path, subset, "Partitions"))

    old_partitions_files = os.listdir(os.path.join(subsets_base_path, subset, "Partitions"))
    
    for subset_file in old_partitions_files:
        if subset_file.endswith(".csv"):
            os.remove(os.path.join(subsets_base_path, subset, "Partitions", subset_file))

    for partition in partitions:
        print(f"Partição {partition}...")
        if partition.count(".zip") == 0:
            csv = pd.read_csv(os.path.join(partitions_base_path, partition))
            csv["Image"].replace(".bmp", ".tif", regex=True, inplace=True)
            csv["Image"].replace("../Datasets/DatasetBalanced", os.path.join(subsets_base_path, subset), regex=True, inplace=True)

            csv.to_csv(os.path.join(subsets_base_path, subset, "Partitions", partition), index=False)