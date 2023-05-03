import os

# Caminho para os datasets
DATASETS_PATH = os.path.join(os.pardir, "Datasets")
OUTPUTCSV_NAME = "NewCases.csv"
removeOutput = True

subsetsDir = os.listdir(DATASETS_PATH)
print("Processing...")
for subset in subsetsDir:
  SUBSET_PATH = os.path.join(DATASETS_PATH, subset)
  classesDir = os.listdir(SUBSET_PATH)

  for className in classesDir:
    CLASS_PATH = os.path.join(SUBSET_PATH, className)
    if os.path.isdir(CLASS_PATH):
      label = className[-1]
      outputFile = os.path.join(SUBSET_PATH, OUTPUTCSV_NAME)
      
      if removeOutput:
        if os.path.isfile(outputFile):
          os.remove(outputFile)
        with open(outputFile, "w") as output:
          output.write(f"ID,Image,Mask,Label,\n")
        removeOutput = False

      imagesList = os.listdir(CLASS_PATH)
      imagesList.remove("masks")
      
      if imagesList.count("ignored.txt") != 0:
        imagesList.remove("ignored.txt")
      
      for image in imagesList:
        maskFile = os.path.join(CLASS_PATH, "masks", image)
        imageFile = os.path.join(CLASS_PATH, image)
        maskPath = os.path.abspath(maskFile)
        imagePath = os.path.abspath(imageFile)
        with open(outputFile, "a") as output:
          output.write(f"{label}-{subset}-{image},{imagePath},{maskPath},{int(label)},\n")

  removeOutput = True
print("Finished")