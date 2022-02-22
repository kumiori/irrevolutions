"""
Skript to plot the angles calculated with Cast3m
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

table=np.genfromtxt("resultantAngles.dat", skip_header=1)
svals=(np.unique(table[:, 0]))

for s in svals:
    plt.figure()
    subtable=table[table[:, 0]==s]
    plt.plot(subtable[:, 1], subtable[:, 4])
    plt.grid()
    plt.title("Resultant angle with different crack length, s="+str(subtable[0, 0])+"mm")
    plt.ylim([-20, 50])
    plt.xlabel("Crack Length, [mm]")
    plt.ylabel("Resultant Angle, [Degrees]")
    plt.savefig("./angleFigs/anglefigs"+str(subtable[0, 0])+".png")
    plt.savefig("./angleFigs/anglefigs"+str(subtable[0, 0])+".pgf")

z=table[:, 4].reshape(len(svals), len(np.unique(table[:, 1])))
plt.figure()
plt.imshow(z, cmap='RdBu', interpolation = 'bilinear', extent=[np.unique(table[:, 1])[0], np.unique(table[:, 1])[-1], np.unique(table[:, 0])[-1], np.unique(table[:, 0])[0]])
plt.colorbar()
plt.savefig("./angleFigs/contourplot.png")
plt.savefig("./angleFigs/contourplot.pgf")