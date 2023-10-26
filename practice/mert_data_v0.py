
import numpy as np
import math
import json
from dolfinx.io import XDMFFile



# f = open('output/time_data.json', 'r')

f = open('output/mert/time_data.json', 'r')

# data = json.load(f)
data = json.loads(f.read())
 
#print(data["load"]) 





load_t = data["load"]
e_total = data["total_energy"]
s_data = data["solver_data"]
sigma_data = data["F"]
f.close()

## --------------------------------------------------------
### Analytical energy
e_a_list = np.zeros(np.size(load_t))
e_a1_list = np.zeros(np.size(load_t))
sigma_a_list = np.zeros(np.size(load_t))
for ind_e in range(np.size(load_t)):
    l_temp = load_t[ind_e]
    if l_temp<=0.8413:
        e_a_list[ind_e] = 0.1*l_temp*l_temp/(2.0 + l_temp*l_temp)
    else:
        e_a_list[ind_e] = 0.1*0.5*0.5/(2.0 + 0.5*0.5)
    sigma_a_list[ind_e] = 0.1*4.0*l_temp/((2.0 + l_temp*l_temp)*(2.0 + l_temp*l_temp))
## --------------------------------------------------------   

import matplotlib.pyplot as plt
plt.plot(load_t, e_total, "k")
plt.plot(load_t, e_a_list, "r--")
plt.xlabel('load')
plt.ylabel('energy')
plt.savefig('output/mert/EvsE.pdf', format='pdf')
plt.close()
plt.plot(load_t, sigma_data, "k")
plt.plot(load_t, sigma_a_list, "r")
plt.savefig('output/mert/SvsS.pdf', format='pdf')
plt.close()



# #print(np.size(load_t), np.size(e_total), np.size(s_data))
# #print(np.size(sigma_data))

# ##
# print('min load: %.6f' %min(load_t))
# print('max load: %.6f' %max(load_t))
# ##



# plt.plot(load_t, sigma_data)
# plt.savefig('output/mert/LvsF.pdf', format='pdf')
# plt.close()

##---------------------------------------------------
## Plot the eigenvalues of Cone
eigs_array = data["cone-eig"]
# print(eigs_array)
counter = 0 
for ind  in range(np.size(load_t)):
    #print(eigs_array[ind])
    if eigs_array[ind] != []:
        counter = counter + 1
        #print(eigs_array[ind])
eig_list = np.zeros(counter)
load_e = np.zeros(counter)
counter = 0 
for ind  in range(np.size(load_t)):
    #print(eigs_array[ind])
    if eigs_array[ind] != []:
        eig_list[counter] = eigs_array[ind]
        load_e[counter] = load_t[ind]
        counter = counter + 1
eig_list = eig_list/max(eig_list)
##
plt.scatter(load_e, eig_list)
plt.xlabel("load")
plt.ylabel("min of eigen values")
plt.ylim([-0.01, 1.01])
plt.savefig('output/mert/cone_eigs.pdf', format='pdf')
plt.close()
##---------------------------------------------------
## Plot the eigenvalues
eigs_array = data["eigs"]
eigs_list = np.zeros(np.size(load_t))
for ind  in range(np.size(load_t)):
    eigs_list[ind] = min(eigs_array[ind][:])
eigs_list = eigs_list/max(eigs_list)
print(np.size(load_t))
print(np.size(eigs_list))
 
e_star = 0.8413
plt.scatter(load_t, eigs_list)
plt.plot([0.0,10], [0.0, 0.0], 'k')
plt.plot([e_star, e_star], [-10.0, 10.0], 'r')
plt.xlabel("load")
plt.ylabel("min of eigen values")
plt.xlim([0.0, 2.0])
plt.ylim([-5.4, 1.4])
plt.savefig('output/mert/eigs.pdf', format='pdf')
plt.close()


#-----------------------------------------------------------
# ## Plot the damage profile
# #
# from test_viz import plot_vector, plot_scalar, plot_profile
# ##
# tol = 1e-3
# xs = np.linspace(0 + tol, Lx - tol, 101)
# points = np.zeros((3, 101))
# points[0] = xs

# _plt, data = plot_profile(
#     u,
#     points,
#     plotter,
#     subplot=(0, 0),
#     lineproperties={
#         "c": "k",
#         "label": f"$u_\ell$ with $\ell$ = {ell:.2f}"
#     },
# )
# ax = _plt.gca()
# ax.axvline(0.0, c="k")
# ax.axvline(2 * ell, c="k", label='D=$2\ell$')
# _plt.legend()
# _plt.fill_between(data[0], data[1].reshape(len(data[1])))
# _plt.title("Variational Inequality")
# _plt.savefig(f"output/mert/test_vi_profile_MPI{MPI.COMM_WORLD.size}-{ell:.3f}.png")





# reader = XdmfReader.New()
# domain = reader.read("Points.xmf")
# grid = domain.getUnstructuredGrid(0)
# topology = grid.getTopology()
# print "Values = ", topology.getValuesString()
# geometry = grid.getGeometry()
# print  "Geo Type =  ", geo.getType().getName(), " #  Points = ", geometry.getNumberOfPoints()
# print  "Points =  ", geometry.getValuesString()


