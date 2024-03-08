import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy as sp
import sys, os, sympy, shutil, math
# import xmltodict
# import pickle
import json
# import pandas
import pylab
from os import listdir
import pandas as pd
import visuals
import hashlib
import yaml

import os.path

print('postproc')
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

elastic = 'C0'
homogen = 'C1'
localis = 'C2'
unstabl = 'C3'

def load_data(rootdir):

    # with open(rootdir + '/parameters.pkl', 'r') as f:
    # 	params = json.load(f)

    with open(rootdir + '/parameters.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    try:
        with open(rootdir + '/time_data.json', 'r') as f:
            data = json.load(f)
            dataf = pd.DataFrame(data).sort_values('load')
        # Continue with your code using the dataf DataFrame
    except FileNotFoundError:
        print("File 'time_data.json' not found. Handle this case accordingly.")
        dataf = pd.DataFrame()
    
    if os.path.isfile(rootdir + '/signature.md5'):
#         print('sig file found')
        with open(rootdir + '/signature.md5', 'r') as f:
            signature = f.read()
    else:
        print('no sig file found')
        signature = hashlib.md5(str(params).encode('utf-8')).hexdigest()

    return params, dataf, signature 

def t_stab(ell, q=2):
    # coeff = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)
    coeff_bif = 2*np.pi*np.sqrt(1/6)
    coeff = coeff_bif/((q+1)/(2.*q))
    if 1/ell > coeff:
#     print(1/ell, coeff)
        return 1.
    else:
        return coeff*ell

def t_bif(ell, q=2):
    # coeff = t_stab(ell, q)*(q+1)/(2.*q)*np.sqrt(2)
    coeff = 2*np.pi*np.sqrt(1/6)
    if 1/ell > coeff:
#     print(1/ell, coeff)
        return 1.
    else:
        return coeff*ell/1

def plot_loadticks(ax, tc, ell):
    if t_stab(ell)-tc < .1:
        # label = '$t_c=t_b=t_s$'
        ax.set_xticks([0,tc])
        ax.set_xticklabels(['0','$t_c$=$t_b$=$t_s$'])
    else:
        ax.set_xticks([0,tc, t_bif(ell), t_stab(ell)])
        ax.set_xticklabels(['0','$t_c$', '$t_b$', '$t_s$'])
    return ax

def plot_fills(ax, ell, tc):
    ax.add_patch(patches.Rectangle((0, 0), 1, 10, facecolor = elastic, fill=True, alpha=.3))
    ax.add_patch(patches.Rectangle((tc, 0), t_bif(ell)-1, 10, facecolor = homogen, fill=True, alpha=.3))
    ax.add_patch(patches.Rectangle((t_bif(ell), 0), t_stab(ell)-t_bif(ell), 10, facecolor = 'w', fill=True, alpha=.3))
    ax.add_patch(patches.Rectangle((t_stab(ell), 0), 10, 10, facecolor = localis, fill=True, alpha=.3))
    return ax

def plot_spectrum(params, data, tc, ax=None, tol=1e-12):
    E0 = params['material']['E']
    w1 = params['material']['sigma_D0']**2/E0
    ell = params['material']['ell']
    fig = plt.figure()
    for i,d in enumerate(data['eigs']):
        if d is not (None and np.inf and np.nan):
            lend = len(d) if isinstance(d, list) else 1
            plt.scatter([(data['load'].values)[i]]*lend, d,
                       c=np.where(np.array(d)<tol, 'red', 'C2'))
                       # c=np.where(np.array(d)<tol, 'C1', 'C2'))

    plt.axhline(0, c='k', lw=2.)
    plt.xlabel('$t$')
    plt.ylabel('Eigenvalues')
    #     plt.ylabel('$$\\lambda_m$$')
    # plt.axvline(tc, lw=.5, c='k')
    ax1 = plt.gca()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    plot_loadticks(ax1, tc, ell)

    ax2 = plt.twinx()
    ax2.plot(data['load'].values, data['alpha_max'].values, label='$$max(\\alpha)$$')
    ax2.legend()
    tbif = t_bif(ell)
    tstab = t_stab(ell)
    ax2.set_ylabel('max $\\alpha$')
    ax2.set_ylim(0, 1.03)
    
    ax = plt.gca()

    # ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
    # ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
    ax.set_xlim(params['loading']['load_min'], params['loading']['load_max'])
    
    ax2.set_yticks([0, 1.])
    ax2.set_yticklabels(['0','1'])
    ax2.set_ylim(0, 1.03)

    # stable = data['stable'].values
    # ax2.scatter(data['load'].values[stable],  -1.+data['stable'].values[stable], c='k', marker='s', s=70, label='stable')
    # ax2.scatter(data['load'].values[~stable],     data['stable'].values[~stable], c='red', marker='s', s=70, label='unstable')

    ax1.tick_params(axis='y', labelrotation=90 )
    plot_fills(ax, ell, tc)

    plt.legend(loc="upper left")


    # ax1.get_yaxis().set_major_formatter(ScalarFormatter())

    return fig, ax1, ax2

def plot_sigmaeps(params, dataf, tc):
    E0 = params['material']['E']
    w1 = params['material']['sigma_D0']**2/E0
    ell = params['material']['ell']
    Lx = params['geometry']['Lx']
    Ly = params['geometry']['Ly']
    
    fig = plt.figure()

    t = np.linspace(0., params['loading']['load_max'], 100)
    fig = plt.figure()
    plt.ylabel('$$\sigma$$')
    plt.xlabel('$$t$$')

    # plt.plot(dataf['load'].values,
        # dataf['load'].values*pow(dataf['S(alpha)'].values, -1), marker='o', label='$$\sigma$$')

    plt.plot(dataf['load'].values,
        dataf['sigma'].values, marker='o', label='$$\sigma$$', c='k', alpha = 0.7)
        # dataf['load'].values*dataf['A(alpha)'].values*E0/Ly, marker='o', label='$$\sigma$$')

    sigmaC = params['material']['sigma_D0']
    ax = plt.gca()
    ax.set_yticks([0, sigmaC])
    ax.set_yticklabels(['0','$$\\sigma_c$$'])
    # ax1.set_yticklabels(['0', '$$10^{-3}$$'])
    ax.set_xticks([0,tc, t_bif(ell), t_stab(ell)])
    ax.set_xticklabels(['0','$t_c$', '$t_b$', '$t_s$'])
    plt.ylim([0, sigmaC*1.1])
    
    stable = dataf['stable'].values

    # ax.axvline(tc, c='k', lw=.5, label='$t^{cr}$')
    # ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
    # ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
    ax.set_xlim(params['loading']['load_min'], params['loading']['load_max'])	

    plot_fills(ax, ell, tc)
    plot_loadticks(ax, tc, ell)

    plt.legend(loc="upper right")

    return fig, ax

def plot_energy(params, dataf, tc):
    E0 = params['material']['E']
    w1 = params['material']['sigma_D0']**2/E0
    ell = params['material']['ell']
    fig = plt.figure()
    Lx = params['geometry']['Lx']
    Ly = params['geometry']['Ly']
    En0 = w1 * Lx * Ly
    t = np.linspace(0., 3., 100)
    fig = plt.figure()
    plt.xlabel('$$t$$')

    dissi = dataf['dissipated_energy'].values
    elast = dataf['elastic_energy'].values
    loads = dataf['load'].values

    plt.plot(loads, dissi/En0, marker='o', label='dissipated', c='white', markeredgewidth=1., markeredgecolor='grey', alpha = 0.7)

    plt.plot(loads, elast/En0, marker='o', label='elastic', c='k', alpha = 0.7)

    plt.plot(loads, (elast+dissi)/En0, marker='o', lw=3, label='total', c='grey', alpha = 0.7)

    ax = plt.gca()
    # ax.axvline(tc, c='k', lw=.5, label='$t^{cr}$')
    # print(ell)
    # print(t_stab(ell))
    # print(t_bif(ell))
    # ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label=r'$t^{cr}_s$')
    # ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
    ax.set_xlim(params['loading']['load_min'], params['loading']['load_max'])
    
    ax.set_ylabel('Energy/$(w_1|\Omega|)$')
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1, 1./2.])
    # ax.set_yticklabels(['$$1$$', '$$1/2$$'])

    plot_loadticks(ax, tc, ell)
    ax.tick_params(axis='y', labelrotation=90 )

    plot_fills(ax, ell, tc)
    plt.legend()

    # ax.add_patch(patches.Rectangle((0, 0), 1, 10, facecolor = elastic, fill=True, alpha=.3))
    # ax.add_patch(patches.Rectangle((tc, 0), t_bif(ell)-1, 10, facecolor = homogen, fill=True, alpha=.3))
    # ax.add_patch(patches.Rectangle((t_bif(ell), 0), t_stab(ell)-t_bif(ell), 10, facecolor = localis, fill=True, alpha=.3))
    # ax.add_patch(patches.Rectangle((1, t_bif(ell)), 10, 10, facecolor = localis,fill=True, alpha=.3))


    return fig, ax

def plot_stability(prefix, tol=1e-5):
    # dirtree = os.path.join(dirroot, signature)
    fig = plt.figure()
    stab_diag = []
    global_dfs = []


    debug = False
    for subdir, dirs, files in os.walk(prefix):
        if not os.path.isfile(subdir + "/parameters.pkl"):
            print('file not found {}'.format(subdir + "/parameters.pkl"))
            continue
        with open(subdir + '/parameters.pkl', 'r') as f: 
            params = json.load(f)
            ell = params['material']['ell']
        if not os.path.isfile(subdir + "/time_data.json"):
            print('file not found {}'.format(subdir + "/time_data.json"))
            continue
        with open(subdir + "/time_data.json") as f:
            data = json.load(f)
            df = pd.DataFrame(data).sort_values('load')
            mineig = [min(eigs) if isinstance(eigs, (list,)) else 100 for eigs in df['eigs']]

            tol = tol
            # print(mineig)
            # print(np.array(mineig) < tol)
            loads = df['load'][np.where(np.array(mineig) < tol)[0]].values
            plt.plot(loads, [1/ell]*len(loads), c='C0', marker='+')
            # label='$\\lambda_{min}<\\eta_{tol}$')
            loads = df['load'][np.where(np.array(mineig) < 0)[0]].values
            plt.plot(loads, [1/ell]*len(loads), c='k', marker='X')
            loads = df['load'][np.where(np.array(mineig) > tol)[0]].values
            # plt.plot(loads, [1/ell]*len(loads), c='C2', marker='.')
            elasticloads = np.where(df['load']<=1)[0]

            # plt.plot(loads, [1/ell]*len(loads), c='C0', marker='.')
            plt.plot(loads[elasticloads[-1]::], [1/ell]*len(loads[elasticloads[-1]::]), c=homogen, marker='.')
            plt.plot(loads[elasticloads], [1/ell]*len(loads[elasticloads]), c=elastic, marker='.')

            if debug:
                print('1/ell, mineog', 1/ell, mineig)
                print('nonunique loads')
                print(1/ell, np.where(np.array(mineig) < tol)[0])
                print('unstable')
                print(1/ell, np.where(np.array(mineig) < 0)[0])


    # plt.plot((20, 20), (20, 20), ls='-', c='C0', marker='+', label='$\\lambda_0<{}$'.format(tol))
    plt.plot((20, 20), (20, 20), ls='', c='k', marker='X', label='incr. unstable')
    plt.plot((20, 20), (20, 20), ls='', c=elastic, marker='.', label='elastic')
    plt.plot((20, 20), (20, 20), ls='', c=homogen, marker='.', label='incr. \\& state stable ')

    q=2
    coeff_sta = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)
    coeff_bif = coeff_sta*(q+1)/(2.*q)
    loads = np.linspace(1., 10., 100)

    ax = plt.gca()
    # ax.plot(loads, [2.*2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)/i for i in loads], lw=3, c='k's)
    ax.plot(loads, [coeff_sta/i for i in loads], '-', c='k', label='$$t_s(L/\ell)$$')
    ax.plot(loads, [coeff_bif/i for i in loads], '-.', c='k', label='$$t_b(L/\ell)$$')
    # plt.axvline(1.0, c='k', lw=1)

    ax.fill_betweenx([coeff_sta/i for i in loads], loads, 20., alpha=.3, facecolor=localis)
    # ax.fill_betweenx([coeff_bif/i for i in loads], 0, loads, alpha=.3, facecolor='C1')
    ax.fill_betweenx([coeff_bif/i for i in loads], 1, loads, alpha=.3, facecolor=homogen)

    # ax.add_patch(patches.Rectangle((0, coeff_bif), 1, 10, facecolor = 'C0',fill=True, alpha=.3))
    ax.add_patch(patches.Rectangle((0, 0), 1, 10, facecolor = elastic,fill=True, alpha=.3))
    ax.add_patch(patches.Rectangle((1, coeff_sta), 10, 10, facecolor = localis,fill=True, alpha=.3))
    x1, y1 = [1, 1], [coeff_bif, 20]
    plt.plot(x1, y1, lw=2, c='k')

    plt.legend(loc='upper right')
    plt.xlabel('$t$')
    plt.ylabel('$$L/\ell$$')
    plt.ylim(0., 1.5*coeff_sta)
    plt.xlim(0., max(loads))

    ax.set_yticks([0, 1, 1/.5, 1/.25, coeff_sta, coeff_bif])
    ax.set_yticklabels(['0','1','2','4', '$$\\ell_s$$', '$$\\ell_b$$'])

    # plt.loglog()
    plt.ylim(0.5, 3*coeff_sta)
    plt.xlim(0.5, max(loads))
    return fig

def load_cont(prefix):
    with open(prefix + '/continuation_data.json', 'r') as f:
        data = json.load(f)
        dataf = pd.DataFrame(data).sort_values('iteration')
    return dataf

def format_params(params):
    return '$$\ell = {:.2f}, \\nu = {:.1f}, \\sigma_c = {:.1f}, ' \
           'E = {:.1f}$$'.format(params['material']['ell'], params['material']['nu'],
                params['material']['sigma_D0'], params['material']['E'])

def _plot_spectrum(data):
    _stab_cnd = lambda data: [0 if data["cone-stable"][i]==True else 1 for i in range(len(data))]
    # _uniq_cnd = [.3 if d[0]>0 else .7 for d in data['eigs']]
    """docstring for plotSpaceVsCone"""
    figure, axis = plt.subplots(1, 1)
    _lambda_0 = [np.nan if type(a) is list else a for a in data["cone-eig"] ]
    
    # __lambda_0 = [e[0] for e in data['eigs']]
    __lambda_0 = [e[0] if len(e)>0 else np.nan for e in data['eigs']]
    scale = __lambda_0[0]
    _colormap=_stab_cnd(data)
    axis.scatter(data.load, np.array(_lambda_0)/scale, c=_colormap, cmap='RdYlGn_r', alpha=.8, s=200, label='cone')
    # axis.scatter(data.load, np.array(__lambda_0)/scale, c=_uniq_cnd, cmap = 'seismic', alpha=.8, label='space')
    axis.scatter(data.load, np.array(__lambda_0)/scale, cmap = 'seismic', alpha=.8, label='space')
    axis.axhline(0., c='k')
    axis.set_xlabel('load')
    axis.set_yticks([0, 1, -3])
    axis.set_ylabel('$min \lambda / \Lambda_0$')
    axis.set_title('Min eig in cone vs. space')
    axis.legend()
    
    return figure, axis

def read_mode_data_from_npz(npz_file, time_step, num_points = -1, num_modes=1):
    """
    Read mode data for a given timestep and x_values from an npz file.

    Parameters:
    - npz_file (numpy.lib.npyio.NpzFile): The npz file containing mode shapes data.
    - time_step (int): The timestep to read.
    - num_modes (int): The number of modes.
    - num_points (int): The number of domain nodes.

    Returns:
    - mode_data (dict): A dictionary containing mode-specific fields for the given timestep.
    """
    mode_data = {}
    mode_data["mesh"] = npz_file["mesh"]
    if 'time_steps' not in npz_file or time_step not in npz_file['time_steps']:
        print(f"No data available for timestep {time_step}.")
        return None

    index = np.where(npz_file['time_steps'] == time_step)[0][0]

    for mode in range(1, num_modes + 1):
        mode_key = f'mode_{mode}'

        if mode_key not in npz_file['point_values'].item():
            print(f"No data available for mode {mode} at timestep {time_step}.")
            continue

        fields = npz_file['point_values'].item()[mode_key]
        if 'bifurcation_β' not in fields or 'stability_β' not in fields:
            print(f"Incomplete data for mode {mode} at timestep {time_step}.")
            continue
        
        field_β_bif_values = np.array(fields['bifurcation_β'][index])
        field_v_bif_values = np.array(fields['bifurcation_v'][index])
        field_β_stab_values = np.array(fields['stability_β'][index])
        field_v_stab_values = np.array(fields['stability_v'][index])

        # Assuming x_values is known or can be obtained
        if num_points == -1:
            num_points = len(npz_file["mesh"])
        x_values = np.linspace(0, 1, num_points)  # Replace with actual x_values
        
        mode_data["fields"] = {
            'bifurcation_β': field_β_bif_values,
            'bifurcation_v': field_v_bif_values,
            'stability_β': field_β_stab_values,
            'stability_v': field_v_stab_values,
        }

        mode_data["time_step"] = time_step
        mode_data["lambda_bifurcation"] = np.nan
        mode_data["lambda_stability"] = np.nan
        
    return mode_data

def plot_fields_for_time_step(mode_shapes_data):
    x_values = mode_shapes_data["mesh"]
    fields = mode_shapes_data["fields"]
    if 'bifurcation_β' in fields and 'stability_β' in fields:
        bifurcation_values = np.array(fields['bifurcation_β'])
        bifurcation_values_v = np.array(fields['bifurcation_v'])
        stability_values = np.array(fields['stability_β'])
        stability_values_v = np.array(fields['stability_v'])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(x_values, bifurcation_values, label='numerical value', marker = 'o')
        axes[0].plot(x_values, bifurcation_values_v, label='numerical value', marker = 'o')
        axes[0].set_title(f'Bifurcation')

        axes[1].plot(x_values, stability_values, label='numerical value', marker = 'o')
        axes[1].plot(x_values, stability_values_v, label='numerical value', marker = 'o')
        axes[1].set_title(f'Stability')

        for axis in axes:
            axis.axhline(0., c='k')
        
        return fig, axes

def plot_operator_spectrum(data, parameters):
    figure, axis = plt.subplots(1, 1)
    scale = data['eigs_ball'].values[0][0]
    tol = parameters["stability"]["cone"]["cone_rtol"]
    colour = np.where(data['eigs_cone'] > tol, 'green', 'red')
    # Concatenate data for all load steps
    load_steps_all = np.concatenate([np.full_like(eigenvalues, load_step) for load_step, eigenvalues in zip(data['load'], data['eigs_ball'])])
    eigenvalues_all = np.concatenate(data['eigs_ball'])

    # Create a scatter plot with vertical alignment
    axis.scatter(load_steps_all, eigenvalues_all / scale, marker='o', c='C0', label='Eigenvalues in vector space')
    axis.scatter(data.load, data['eigs_cone'], marker = 'd', c=colour, s=60, label='Eigenvalues in cone')

    axis.set_xlabel(r'Load $t$')
    axis.set_ylabel('Eigenvalues')
    axis.set_title('Spectrum of Nonlinear Operator $H_{\\ell}:=E_\\ell\'\'(y_t)$')
    axis.set_yticks([0, 1, -3])
    # axis.axhline(0., c='k')
    axis.axhline(tol, c='k')
    axis.set_ylim(-.5, 1.5)
    axis.grid(True)
    axis.legend()
    
    return figure, axis
    # axis.show()
