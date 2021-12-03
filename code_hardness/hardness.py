import json
from pymatgen import Structure
from matminer.featurizers.composition import Meredig
import numpy as np

target = "Hardness"

with open('../data.json') as json_file:
    data = json.load(json_file)


structure = Structure.from_dict(data[0])
hardness = data[0]["hardness"]


feature_list=[]
label_list=[]

for i in range(len(data)):
  s = Structure.from_dict(data[i])
  F = Meredig().featurize(s.composition)
  comp_features = [F[4], F[6], F[7], F[103], F[104], F[105], F[106], F[107],\
                   F[108], F[109], F[110], F[111], F[112], F[113], F[116], F[117], data[i]["density"] ]
  feature_list.append(comp_features)
  label_list.append(data[i]["hardness"])

#### start ML #####

print("num_of_samples : ", len(feature_list))
X=np.array(feature_list)
y=np.array(label_list)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

indices = np.arange(len(feature_list))

# split data into train+validation set and test set 
X_trainval, X_test, y_trainval, y_test, I_X_trainval, I_X_test = train_test_split(
    X, y, indices, random_state=0, test_size=0.2)


# split train+validation set into training and validation sets 
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)


# sklearn GridSearchCV:

param_grid = {'max_depth': [6], 'n_estimators':[100], 'n_jobs':[-1]}
print("Parameter grid:\n{}".format(param_grid))


grid_search = GridSearchCV(RandomForestRegressor(random_state=3), param_grid, cv=10)
tree = grid_search.fit(X_trainval, y_trainval)
# save model using joblib
import joblib
joblib.dump(tree, 'rf_hardness.joblib')

pred_tree = tree.predict(X_test)

print('')
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
print("Test set score: {:.3f}".format(grid_search.score(X_test, y_test)))
print("Test set Pearson score: {:.3f}".format(np.corrcoef(pred_tree, y_test)[0, 1]))

plt.plot([0,100],[0,100],color='r',linestyle='dashed')
plt.scatter(pred_tree, y_test,edgecolors='b',alpha=0.4 )
plt.xlabel("Predicted value (GPa)", fontsize=12)
plt.ylabel("DFT value (GPa)", fontsize=12)
plt.text(18,45,target, fontsize=12)
plt.text(32,10,"$r$ = {:.3f}".format(np.corrcoef(pred_tree, y_test)[0, 1]), fontsize=12)
plt.xlim((0.0, 50.0))
plt.ylim((0.0, 50.0))

ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

plt.axes().set_aspect('equal')
plt.savefig('evaluation_hardness.png',dpi=300,bbox_inches='tight')
plt.clf()



from sklearn import datasets
import pandas as pd

feature_names=['B fraction', 'N fraction', 'O fraction', 'mean AtomicWeight', 'mean Column', 'mean Row', 'range Number', 'mean Number', 'range AtomicRadius', 'mean AtomicRadius', 'range Electronegativity', 'mean Electronegativity', 'avg s valence electrons', 'avg p valence electrons', 'frac s valence electrons', 'frac p valence electrons', 'density']

df = pd.DataFrame(data = feature_list, columns = feature_names)
df.to_csv('features_hardness.csv', sep = ',', index = False)

#######################################################################################################

# feature importance

importances = grid_search.best_estimator_.feature_importances_
included = feature_names
indices = np.argsort(importances)[::-1]

sorted_idx = grid_search.best_estimator_.feature_importances_.argsort()#.tolist()
sorted_idx = sorted_idx[::-1]
sorted_idx = sorted_idx[0:5]
#print("sorted index")
#print(sorted_idx[0:5])
#print(type(sorted_idx))
#print(feature_names[sorted_idx])
#print(type(grid_search.best_estimator_.feature_importances_))
#print(grid_search.best_estimator_.feature_importances_[sorted_idx])
#print(grid_search.best_estimator_.feature_importances_)
plt.barh(np.array(feature_names)[sorted_idx], grid_search.best_estimator_.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig('feature_importances_hardness.png',dpi=300,bbox_inches='tight')
plt.clf()

