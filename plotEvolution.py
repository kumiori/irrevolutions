import numpy as np
import matplotlib.pyplot as plt
from pip import main

plotPath="plots/s10_fine/alpha7600_c.png"
x0=210
y0=294
x1=233
y1=264

def rootFunction(x):
    a=(y1-y0)/np.sqrt(x1-x0)
    return a*np.sqrt(x-x0)+y0

def main():
    x=np.linspace(x0, x1, 100)
    y=rootFunction(x)
    img = plt.imread(plotPath)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(x, y, linewidth=1, color="r")
    ax.axis('off')
    plt.show()
    fig.savefig("./alpha_part_parabola.png", transparent=True,bbox_inches='tight')
    print("Gamma= "+str((263-x0)/(y0-268)))



if __name__=="__main__":
    main()