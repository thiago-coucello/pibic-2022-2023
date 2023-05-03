

# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# load data
ROOT = 'G:\My Drive\Doutorado\BaseColuna\pyradiomics'
filename = os.path.join(ROOT, 'Fraturado OU Nao', 'featuresT1.csv')
dataframe = read_csv(filename)
array = dataframe.values


X = array[:,0:1070]
Y = array[:,1070]
# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])