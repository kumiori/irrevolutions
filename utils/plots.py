import numpy as np
import matplotlib.pyplot


def plot_energies(history_data, title="Evolution", file=None):

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Energies", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    e_e = np.array(history_data["elastic_energy"])
    e_d = np.array(history_data["dissipated_energy"])

    # stress-strain curve
    ax1.plot(
        t,
        e_e,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Elastic",
    )
    ax1.plot(
        t,
        e_d,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="^",
        label=r"Dissipated",
    )
    ax1.plot(t, e_d + e_e, color="black", linestyle="-", linewidth=1.0, label=r"Total")

    ax1.legend(loc="upper left")
    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_AMit_load(history_data, title="AM max it - Load", file=None):

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"AM max it", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    it = np.zeros_like(t)
    for i, load in enumerate(t):
        it[i] = np.array(history_data["solver_data"][i]["iteration"][-1])

    # stress-strain curve
    ax1.plot(
        t,
        it,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=2.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_force_displacement(history_data, title="Force - Displacement", file=None):

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Force", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    F = np.array(history_data["F"])

    # stress-strain curve
    ax1.plot(
        t,
        F,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_residual_AMit(
    history_data, load_check, criterion, title="Residual - AM it", file=None
):

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Residual", fontsize=12)
    ax1.set_ylabel(r"AM it", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    it = np.array(history_data["solver_data"][load_check]["iteration"])
    if criterion == "residaul_u":
        R = np.array(history_data["solver_data"][load_check]["error_residual_u"])
    if criterion == "alpha_H1":
        R = np.array(history_data["solver_data"][load_check]["error_alpha_H1"])

    # stress-strain curve
    ax1.plot(
        it,
        R,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_energy_AMit(history_data, load_check, title="Total energy - AM it", file=None):

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Energy", fontsize=12)
    ax1.set_ylabel(r"AM it", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    it = np.array(history_data["solver_data"][load_check]["iteration"])
    E = np.array(history_data["solver_data"][load_check]["total_energy"])

    # stress-strain curve
    ax1.plot(
        it,
        E,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1
