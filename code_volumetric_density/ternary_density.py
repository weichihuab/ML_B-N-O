
import joblib
from pymatgen import Structure
from matminer.featurizers.composition import Meredig
import numpy as np
import matplotlib.pyplot as plt
import ternary
import matplotlib as mpl
mpl.use('Agg')


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
        pred_feature_list.append(comp_features)

        pred_X=np.array(pred_feature_list)
        pred_y=tree.predict(pred_X)
        d[(i,j)] = pred_y[0]
    return d

scale = 20

tree = joblib.load("rf_density.joblib")
ternary_data = func1(scale)
figure, tax = ternary.figure(scale=scale)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)

# Set Axis labels and Title
fontsize = 14
fontweight = 'bold'
tax.right_corner_label("B", fontsize=fontsize, fontweight=fontweight)
tax.top_corner_label("N", fontsize=fontsize, fontweight=fontweight)
tax.left_corner_label("O", fontsize=fontsize, fontweight=fontweight)

cb_kwargs = {'shrink': 1.0}
tax.heatmap(ternary_data, style="hexagonal", cmap='pink_r', cbarlabel=r'Predicted density (atom/Å$^3$)', scientific=False, cb_kwargs=cb_kwargs)


tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.boundary()

tax.show()
figure.savefig('ternary_density.png',dpi=300)


