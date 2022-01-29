import numpy as np
import matplotlib.pyplot as plt

def main(filename="energyEll.txt"):
    """ Plot the resulting development of the minimum energy
    Input:
    Reads in energy devolpment in txt format, standard filename "energyEll.txt"
    Output:
    Shows plot and saves it as png file (can be adapted for publications to LaTeX Compatible formats)
    """
    data=np.genfromtxt(filename)
    plt.plot(data[0, :], data[1, :])
    plt.scatter(data[0, :], data[1, :])
    plt.title("Energy")
    plt.ylabel("Energy")
    plt.xlabel("Ell")
    plt.grid()
    plt.savefig("energyEll.png")
    plt.show()


if __name__=="__main__":
    main()