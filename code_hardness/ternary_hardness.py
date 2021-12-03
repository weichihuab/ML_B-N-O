
import os
from matminer.featurizers.composition import Meredig
from pymatgen import Structure
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

  
#########################################################################################################

import ternary
import joblib

tree = joblib.load("rf_hardness.joblib")
density_tree = joblib.load("../code_volumetric_density/rf_density.joblib")

def func1(scale=5):
    from ternary.helpers import simplex_iterator
    from pymatgen import Composition
    d = dict()
    for (i,j,k) in simplex_iterator(scale):
        #name='B'+str(i)+' N'+str(j)+' O'+str(k)
        name = Composition({'B': i, 'N': j, 'O': k})

        pred_feature_list=[]
        F = Meredig().featurize(name)
        comp_features = [F[4], F[6], F[7], F[103], F[104], F[105], F[106], F[107],\
                         F[108], F[109], F[110], F[111], F[112], F[113], F[116], F[117] ]


        feature_list_1 = [comp_features]
        density_X=np.array(feature_list_1)
        density_y=density_tree.predict(density_X)
        all_features=[]
        all_features.append(comp_features+list(density_y))
        #print("all_features",all_features)

        pred_X=np.array(all_features)
        pred_y=tree.predict(pred_X)
        d[(i,j)] = pred_y[0]
    return d

scale = 20
ternary_data = func1(scale)

#print(ternary_data)  # a dictionary {(0, 0): 7.671940764488272, (0, 1): 38.67253144530925, (0, 2): 40.41370563109955,  ....
                      # (18, 0): 25.76023989668987, (18, 1): 25.683451835332313, (18, 2): 25.375351154434078, (19, 0): 25.36082059366725, (19, 1): 30.325333491271113
                      #  , (20, 0): 33.925398545236504}                 

figure, tax = ternary.figure(scale=scale)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
#tax.gridlines(color="blue", multiple=5)

# Set Axis labels and Title
fontsize = 14
fontweight = 'bold'
#tax.set_title("Various Lines\n", fontsize=fontsize)
tax.right_corner_label("B", fontsize=fontsize, fontweight=fontweight)
tax.top_corner_label("N", fontsize=fontsize, fontweight=fontweight)
tax.left_corner_label("O", fontsize=fontsize, fontweight=fontweight)

cb_kwargs = {'shrink': 1.0}
tax.heatmap(ternary_data, style="hexagonal", cmap='OrRd', cbarlabel=r'Predicted Hardness (GPa)', scientific=False, cb_kwargs=cb_kwargs, vmax=50.0, vmin=0.0)


#tax.ticks(clockwise=True)

tax.boundary(linewidth=2.0)
#tax.set_title("Bulk modulus (GPa)")


tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.boundary()

tax.show()
figure.savefig("ternary_hardness.png",dpi=300)


