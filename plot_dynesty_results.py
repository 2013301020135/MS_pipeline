#!/usr/bin/env python
"""Analyze dynesty results, extract posteriors and evidences, draw plots for bin selection.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import pickle
import corner
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from dynesty import plotting
# from dynesty import results
# from dynesty import utils as dyfunc


def get_bin_num(**kwargs):
    """Get the bin numbers of noise model"""
    model_components = ""
    for noisebin in kwargs['modelname'].split('+'):
        if 'RN' in noisebin:
            kwargs['RN'] = True
            model_components += "+RN"
            kwargs['rnbin'] = int(noisebin[3:]) if 'b' in noisebin else None
            kwargs['rnc'] = None if 'b' in noisebin else int(noisebin[2:])
        elif 'DM' in noisebin:
            kwargs['DM'] = True
            model_components += "+DM"
            kwargs['dmbin'] = int(noisebin[3:]) if 'b' in noisebin else None
            kwargs['dmc'] = None if 'b' in noisebin else int(noisebin[2:])
        elif 'SV' in noisebin:
            kwargs['SV'] = True
            model_components += "+SV"
            kwargs['svbin'] = int(noisebin[3:]) if 'b' in noisebin else None
            kwargs['svc'] = None if 'b' in noisebin else int(noisebin[2:])
        elif 'SW' in noisebin:
            kwargs['SW'] = True
            model_components += "+SW"
            kwargs['swbin'] = int(noisebin[3:]) if 'b' in noisebin else None
            kwargs['swc'] = None if 'b' in noisebin else int(noisebin[2:])
    kwargs['model'] = model_components[1:]
    return kwargs


def copy_files(**kwargs):
    """Copy and rename the results files, parameter names files, and prior files.

    :return: pkl_file: the pkl results file
    :return: param_file: the parameter names file
    :return: prior_file: the priors file
    """
    pkl_file = f"{kwargs['datadir']}/{kwargs['outdir']}/{kwargs['filename']}.pkl"
    param_file = f"{kwargs['datadir']}/{kwargs['outdir']}/{kwargs['filename']}_pars.txt"
    prior_file = f"{kwargs['datadir']}/{kwargs['outdir']}/{kwargs['filename']}_priors.txt"
    if os.path.exists(f"{kwargs['datadir']}/{kwargs['outdir']}/DynRes.pkl"):
        shutil.copy(f"{kwargs['datadir']}/{kwargs['outdir']}/DynRes.pkl", pkl_file)
    if os.path.exists(f"{kwargs['datadir']}/{kwargs['outdir']}/pars.txt"):
        shutil.copy(f"{kwargs['datadir']}/{kwargs['outdir']}/pars.txt", param_file)
    if os.path.exists(f"{kwargs['datadir']}/{kwargs['outdir']}/priors.txt"):
        shutil.copy(f"{kwargs['datadir']}/{kwargs['outdir']}/priors.txt", prior_file)
    return pkl_file, param_file, prior_file


def load_dynesty_results(pklf):
    """Load dynesty results DynRes.pkl file.

    :param pklf: the pkl file containing the chains

    :return: pklres: the pkl results after loading with pickle
    """
    with open(pklf, 'rb') as f:
        plkres = pickle.load(f)
    return plkres


def safe_weights(pklres):
    """Calculating normalized weights.

    :param pklres: the pkl results after loading with pickle

    :return: weights: the weights relative to evidence
    :return: samples: the samples after resample
    :return: normweights: the weights after normalizing
    """
    logwt = pklres.logwt - np.max(pklres.logwt)
    normweights = np.exp(logwt)
    normweights /= np.sum(normweights)
    samples = plotting.resample_equal(pklres.samples, normweights)
    return samples, normweights


def get_stats(samps):
    """Calculate the statistics of samples"""
    ul68 = np.percentile(samps, 84.1, axis=0)
    ll68 = np.percentile(samps, 15.9, axis=0)
    ul = np.percentile(samps, 97.5, axis=0)
    ll = np.percentile(samps, 2.5, axis=0)
    ulm = np.max(samps, axis=0)
    llm = np.min(samps, axis=0)
    means = np.mean(samps, axis=0)
    stds = np.std(samps, axis=0)
    medians = np.median(samps, axis=0)
    return means, stds, llm, ll, ll68, medians, ul68, ul, ulm


def get_dynesty_posteriors(pklres, paranames, **kwargs):
    """Get the posterirors of parameters from dynesty results.

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    burn = kwargs['burn']
    if burn < 0:
        burn = 0
    elif burn > 1:
        burn = 1
    samples, normweights = safe_weights(pklres)
    n_total = len(pklres.samples)
    n_burn = int(n_total * burn)
    samples_burned = pklres.samples[n_burn:]
    weights_burned = normweights[n_burn:]
    logl_burned = pklres.logl[n_burn:]
    samples_equal = plotting.resample_equal(samples_burned, weights_burned)
    ms, stds, llm, ll97, ll68, mds, ul68, ul97, ulm = get_stats(samples_equal)
    map_imax = np.argmax(weights_burned)
    mpmax = samples_burned[map_imax, :]
    maxlh_imax = np.argmax(logl_burned)
    mlhmax = samples_burned[maxlh_imax, :]
    posterior = f"{kwargs['datadir']}/{kwargs['outdir']}/{kwargs['filename']}_posterior.txt"
    with open(posterior, "w") as resf:
        form = "{:25s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}"
        s = form.format("Parameters", "MAP", "max-like", "mean", "std", "2.5%", "15.9%", "50%", "84.1%", "97.5%")
        resf.write(s + "\n")
        for p, v, mp, m, std, l, ll, md, ul, u in zip(paranames, mlhmax, mpmax, ms, stds, ll97, ll68, mds, ul68, ul97):
            form = "{:25s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}"
            s = form.format(p, v, mp, m, std, l, ll, md, ul, u)
            resf.write(s + "\n")


def load_posteriors(posts):
    """Load the posteriors for RN/DM/SV/SW from posteriors.txt.

    :param posts: the text files containing the posteriors

    :return: p: the dictionary containing the posteriors
    """
    p = {}
    with open(posts, "r") as f:
        for line in f:
            if "rn_log10_A" in line:
                e = line.split()
                p['RN log10amp MAP'], p['RN log10amp max-like'], p['RN log10amp mean'] = e[1], e[2], e[3]
                p['RN log10amp std'], p['RN log10amp 03'], p['RN log10amp 16'] = e[4], e[5], e[6]
                p['RN log10amp 50'], p['RN log10amp 84'], p['RN log10amp 97'] = e[7], e[8], e[9]
            elif "rn_gamma" in line:
                e = line.split()
                p['RN gamma MAP'], p['RN gamma max-like'], p['RN gamma mean'] = e[1], e[2], e[3]
                p['RN gamma std'], p['RN gamma 03'], p['RN gamma 16'] = e[4], e[5], e[6]
                p['RN gamma 50'], p['RN gamma 84'], p['RN gamma 97'] = e[7], e[8], e[9]
            elif "rn_k_dropbin" in line:
                e = line.split()
                p['RN Ncoeff MAP'], p['RN Ncoeff max-like'], p['RN Ncoeff mean'] = e[1], e[2], e[3]
                p['RN Ncoeff std'], p['RN Ncoeff 03'], p['RN Ncoeff 16'] = e[4], e[5], e[6]
                p['RN Ncoeff 50'], p['RN Ncoeff 84'], p['RN Ncoeff 97'] = e[7], e[8], e[9]
            elif "dm_gp_log10_A" in line:
                e = line.split()
                p['DM log10amp MAP'], p['DM log10amp max-like'], p['DM log10amp mean'] = e[1], e[2], e[3]
                p['DM log10amp std'], p['DM log10amp 03'], p['DM log10amp 16'] = e[4], e[5], e[6]
                p['DM log10amp 50'], p['DM log10amp 84'], p['DM log10amp 97'] = e[7], e[8], e[9]
            elif "dm_gp_gamma" in line:
                e = line.split()
                p['DM gamma MAP'], p['DM gamma max-like'], p['DM gamma mean'] = e[1], e[2], e[3]
                p['DM gamma std'], p['DM gamma 03'], p['DM gamma 16'] = e[4], e[5], e[6]
                p['DM gamma 50'], p['DM gamma 84'], p['DM gamma 97'] = e[7], e[8], e[9]
            elif "dm_gp_k_dropbin" in line:
                e = line.split()
                p['DM Ncoeff MAP'], p['DM Ncoeff max-like'], p['DM Ncoeff mean'] = e[1], e[2], e[3]
                p['DM Ncoeff std'], p['DM Ncoeff 03'], p['DM Ncoeff 16'] = e[4], e[5], e[6]
                p['DM Ncoeff 50'], p['DM Ncoeff 84'], p['DM Ncoeff 97'] = e[7], e[8], e[9]
            elif "sv_gp_log10_A" in line:
                e = line.split()
                p['SV log10amp MAP'], p['SV log10amp max-like'], p['SV log10amp mean'] = e[1], e[2], e[3]
                p['SV log10amp std'], p['SV log10amp 03'], p['SV log10amp 16'] = e[4], e[5], e[6]
                p['SV log10amp 50'], p['SV log10amp 84'], p['SV log10amp 97'] = e[7], e[8], e[9]
            elif "sv_gp_gamma" in line:
                e = line.split()
                p['SV gamma MAP'], p['SV gamma max-like'], p['SV gamma mean'] = e[1], e[2], e[3]
                p['SV gamma std'], p['SV gamma 03'], p['SV gamma 16'] = e[4], e[5], e[6]
                p['SV gamma 50'], p['SV gamma 84'], p['SV gamma 97'] = e[7], e[8], e[9]
            elif "sv_gp_k_dropbin" in line:
                e = line.split()
                p['SV Ncoeff MAP'], p['SV Ncoeff max-like'], p['SV Ncoeff mean'] = e[1], e[2], e[3]
                p['SV Ncoeff std'], p['SV Ncoeff 03'], p['SV Ncoeff 16'] = e[4], e[5], e[6]
                p['SV Ncoeff 50'], p['SV Ncoeff 84'], p['SV Ncoeff 97'] = e[7], e[8], e[9]
            elif "sw_gp_log10_A" in line:
                e = line.split()
                p['SW log10amp MAP'], p['SW log10amp max-like'], p['SW log10amp mean'] = e[1], e[2], e[3]
                p['SW log10amp std'], p['SW log10amp 03'], p['SW log10amp 16'] = e[4], e[5], e[6]
                p['SW log10amp 50'], p['SW log10amp 84'], p['SW log10amp 97'] = e[7], e[8], e[9]
            elif "sw_gp_gamma" in line:
                e = line.split()
                p['SW gamma MAP'], p['SW gamma max-like'], p['SW gamma mean'] = e[1], e[2], e[3]
                p['SW gamma std'], p['SW gamma 03'], p['SW gamma 16'] = e[4], e[5], e[6]
                p['SW gamma 50'], p['SW gamma 84'], p['SW gamma 97'] = e[7], e[8], e[9]
            elif "sw_gp_k_dropbin" in line:
                e = line.split()
                p['SW Ncoeff MAP'], p['SW Ncoeff max-like'], p['SW Ncoeff mean'] = e[1], e[2], e[3]
                p['SW Ncoeff std'], p['SW Ncoeff 03'], p['SW Ncoeff 16'] = e[4], e[5], e[6]
                p['SW Ncoeff 50'], p['SW Ncoeff 84'], p['SW Ncoeff 97'] = e[7], e[8], e[9]
    return p


def extract_evidence(pklres, **kwargs):
    """Extract evidence and posteriors of models with fixed bin numbers to evidence.csv.

    :param pklres: the pkl results after loading with pickle
    """
    entry = []
    psr_idx = ['Pulsar', 'Model', 'Bin selection', 'RN', 'DM', 'SV', 'SW', 'RN bin', 'DM bin', 'SV bin', 'SW bin',
               'burn', 'nlive', 'niter', 'ncall', 'efficiency',
               'log weights', 'log prior', 'log likelihood', 'log evidence', 'log evidence std']
    bin_idx = ['RN log10amp MAP', 'RN log10amp max-like', 'RN log10amp mean', 'RN log10amp std',
               'RN log10amp 03', 'RN log10amp 16', 'RN log10amp 50', 'RN log10amp 84', 'RN log10amp 97',
               'RN gamma MAP', 'RN gamma max-like', 'RN gamma mean', 'RN gamma std',
               'RN gamma 03', 'RN gamma 16', 'RN gamma 50', 'RN gamma 84', 'RN gamma 97',
               'RN Ncoeff MAP', 'RN Ncoeff max-like', 'RN Ncoeff mean', 'RN Ncoeff std',
               'RN Ncoeff 03', 'RN Ncoeff 16', 'RN Ncoeff 50', 'RN Ncoeff 84', 'RN Ncoeff 97',
               'DM log10amp MAP', 'DM log10amp max-like', 'DM log10amp mean', 'DM log10amp std',
               'DM log10amp 03', 'DM log10amp 16', 'DM log10amp 50', 'DM log10amp 84', 'DM log10amp 97',
               'DM gamma MAP', 'DM gamma max-like', 'DM gamma mean', 'DM gamma std',
               'DM gamma 03', 'DM gamma 16', 'DM gamma 50', 'DM gamma 84', 'DM gamma 97',
               'DM Ncoeff MAP', 'DM Ncoeff max-like', 'DM Ncoeff mean', 'DM Ncoeff std',
               'DM Ncoeff 03', 'DM Ncoeff 16', 'DM Ncoeff 50', 'DM Ncoeff 84', 'DM Ncoeff 97',
               'SV log10amp MAP', 'SV log10amp max-like', 'SV log10amp mean', 'SV log10amp std',
               'SV log10amp 03', 'SV log10amp 16', 'SV log10amp 50', 'SV log10amp 84', 'SV log10amp 97',
               'SV gamma MAP', 'SV gamma max-like', 'SV gamma mean', 'SV gamma std',
               'SV gamma 03', 'SV gamma 16', 'SV gamma 50', 'SV gamma 84', 'SV gamma 97',
               'SV Ncoeff MAP', 'SV Ncoeff max-like', 'SV Ncoeff mean', 'SV Ncoeff std',
               'SV Ncoeff 03', 'SV Ncoeff 16', 'SV Ncoeff 50', 'SV Ncoeff 84', 'SV Ncoeff 97',
               'SW log10amp MAP', 'SW log10amp max-like', 'SW log10amp mean', 'SW log10amp std',
               'SW log10amp 03', 'SW log10amp 16', 'SW log10amp 50', 'SW log10amp 84', 'SW log10amp 97',
               'SW gamma MAP', 'SW gamma max-like', 'SW gamma mean', 'SW gamma std',
               'SW gamma 03', 'SW gamma 16', 'SW gamma 50', 'SW gamma 84', 'SW gamma 97',
               'SW Ncoeff MAP', 'SW Ncoeff max-like', 'SW Ncoeff mean', 'SW Ncoeff std',
               'SW Ncoeff 03', 'SW Ncoeff 16', 'SW Ncoeff 50', 'SW Ncoeff 84', 'SW Ncoeff 97']
    sum_idx = psr_idx + bin_idx
    bin_sel = False if kwargs['evidence'] else True
    rnbin = kwargs['rnc'] if kwargs['evidence'] else kwargs['rnbin']
    dmbin = kwargs['dmc'] if kwargs['evidence'] else kwargs['dmbin']
    svbin = kwargs['svc'] if kwargs['evidence'] else kwargs['svbin']
    swbin = kwargs['swc'] if kwargs['evidence'] else kwargs['swbin']
    s = {'Pulsar': kwargs['psrname'], 'Model': kwargs['model'], 'Bin selection': bin_sel, 'burn': kwargs['burn'],
         'RN': kwargs['RN'], 'DM': kwargs['DM'], 'SV': kwargs['SV'], 'SW': kwargs['SW'],
         'RN bin': rnbin, 'DM bin': dmbin, 'SV bin': svbin, 'SW bin': swbin,
         'nlive': pklres.nlive, 'niter': pklres.niter, 'ncall': np.sum(pklres.ncall), 'efficiency': pklres.eff,
         'log weights': pklres.logwt[-1], 'log prior': pklres.logvol[-1], 'log likelihood': pklres.logl[-1],
         'log evidence': pklres.logz[-1], 'log evidence std': pklres.logzerr[-1]}
    posts = f"{kwargs['datadir']}/{kwargs['outdir']}/{kwargs['filename']}_posterior.txt"
    sum_slt = {**s, **load_posteriors(posts)}
    series_slt = pd.Series(sum_slt, index=sum_idx)
    entry.append(series_slt)
    frame_slt = pd.DataFrame(entry)
    suffix = "_evidences.csv" if kwargs['evidence'] else "_bin_evidences.csv"
    fnlrslt = f"{kwargs['datadir']}/{kwargs['psrname']}{suffix}"
    if os.path.exists(fnlrslt):
        frame_slt.to_csv(fnlrslt, mode='a', index=False, header=False)
    else:
        frame_slt.to_csv(fnlrslt, mode='w', index=False, header=True)


def write_summary(pklres, **kwargs):
    """Write the summary of the run to summary.txt.

    :param pklres: the pkl results after loading with pickle
    """
    print(pklres.summary())
    with open(f"{kwargs['datadir']}/{kwargs['outdir']}/summary_{kwargs['filename']}.txt", "w") as sf:
        sf.writelines(f"Pulsars: {kwargs['psrname']} \n")
        sf.writelines(f"Noise model: {kwargs['modelname']} \n")
        sf.writelines(f"Red Noise model: components - {kwargs['rnc']}, bin number - {kwargs['rnbin']}\n")
        sf.writelines(f"Dispersion Measure model: components - {kwargs['dmc']}, bin number - {kwargs['dmbin']}\n")
        sf.writelines(f"Scattering Variation model: components - {kwargs['svc']}, bin number - {kwargs['svbin']}\n")
        sf.writelines(f"Solar wind model: components - {kwargs['swc']}, bin number - {kwargs['swbin']}\n")
        sf.writelines(f"The number of live points used in the run: {pklres.nlive} \n")
        sf.writelines(f"The number of iterations (samples): {pklres.niter} \n")
        sf.writelines(f"The total number of function calls: {np.sum(pklres.ncall)} \n")
        sf.writelines(f"The overall sampling efficiency: {pklres.eff}% \n")
        sf.writelines(f"The final cumulative log-evidence with error: {pklres.logz[-1]} +/- {pklres.logzerr[-1]}\n")
        sf.writelines(f"The final log-likelihood: {pklres.logl[-1]} \n")
        sf.writelines(f"The final log-weights: {pklres.logwt[-1]} \n")
        sf.writelines(f"The final (expected) log (prior volume): {pklres.logvol[-1]} \n")
        sf.writelines(f"The final estimated log information: {pklres.information[-1]} \n")


def check_dynesty_data_quality(pklres, **kwargs):
    """Check dynesty data. (Not used)

    :param pklres: the pkl results after loading with pickle
    """
    logz = pklres.logz
    print(f"logz range: {np.nanmin(logz)} to {np.nanmax(logz)}")
    print(f"logz include NaN: {np.isnan(logz).any()}")
    print(f"logz include Inf: {np.isinf(logz).any()}")
    logwt = pklres.logwt
    print(f"logwt range: {np.nanmin(logwt)} to {np.nanmax(logwt)}")
    print(f"logwt include NaN: {np.isnan(logwt).any()}")
    print(f"logwt include Inf: {np.isinf(logwt).any()}")

    for i in range(res.samples.shape[1]):
        samples = res.samples[:, i]
        print(f"Parameter {i} range: {np.nanmin(samples):.4e} to {np.nanmax(samples):.4e}")

    samples, normweights = safe_weights(pklres)
    ess = 1 / np.sum(normweights ** 2)
    print(f"Effective samples: {ess:.1f}/{len(normweights)}")


def plot_parameter_traces(pklres, paranames, **kwargs):
    """Plot parameter traces. (Not used)

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    n_params = len(paranames)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(paranames):
        ax = axes[i]
        ax.plot(pklres.samples[:, i], 'k-', alpha=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter Value")
        ax.grid(alpha=0.3)
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
    
    plt.suptitle("Parameter Traces", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("parameter_traces.png", dpi=300)
    plt.close()
    print("Saved parameter_traces.png")


def plot_parameter_histograms(pklres, paranames, **kwargs):
    """Plot parameter histograms. (Not used)

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    n_params = len(paranames)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    samples, normweights = safe_weights(pklres)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()

    for i, name in enumerate(paranames):
        ax = axes[i]
        ax.hist(samples[:, i], bins=50, density=True, 
                histtype='step', color='blue', alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Probability Density")
        ax.grid(alpha=0.3)
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
    
    plt.suptitle("Parameter Distributions", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("parameter_histograms.png", dpi=300)
    plt.close()
    print("Saved parameter_histograms.png")


def plot_corner(pklres, paranames, **kwargs):
    """Plot corner plots.

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    samples, normweights = safe_weights(pklres)
    lh = pklres.logl
    fig = corner.corner(samples, labels=paranames, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 12}, plot_datapoints=False,
                        fill_contours=True, levels=[0.68, 0.95], smooth=1.0)
    # fig = corner.corner(pklres.samples, weights=normweights, labels=paranames, quantiles=[0.16, 0.5, 0.84],
    #                     show_titles=True, title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 12},
    #                     plot_datapoints=False, fill_contours=True, levels=[0.68, 0.95], smooth=1.0)
    plt.suptitle("Posterior Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{kwargs['outdir']}/cornerplot_{kwargs['filename']}.png", dpi=300)
    plt.close()
    print(f"Saved corner plot: {kwargs['outdir']}/cornerplot_{kwargs['filename']}.png")


def plot_run(pklres, paranames, **kwargs):
    """Plot run plots.

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    if not hasattr(pklres, 'nlive') or not pklres.nlive:
        print("Warning: no nlive data available")
        return
    fig, axes = plotting.runplot(pklres, labels=paranames, logplot=True)
    plt.savefig(f"{kwargs['outdir']}/runplot_{kwargs['filename']}.png", dpi=300)
    plt.close()
    print(f"Saved run plot: {kwargs['outdir']}/runplot_{kwargs['filename']}.png")


def plot_trace(pklres, paranames, **kwargs):
    """Plot trace plots.

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    fig, axes = plotting.traceplot(pklres, labels=paranames, show_titles=True, trace_cmap='viridis', connect=True)
    plt.savefig(f"{kwargs['outdir']}/traceplot_{kwargs['filename']}.png", dpi=300)
    plt.close()
    print(f"Saved trace plot: {kwargs['outdir']}/traceplot_{kwargs['filename']}.png")
    

def plot_bound(pklres, paranames, **kwargs):
    """Plot bound plots.

    :param pklres: the pkl results after loading with pickle
    :param paranames: the names of parameters in the chains
    """
    if not hasattr(pklres, 'bound') or not pklres.bound:
        print("Warning: no bound data available")
        return
    param_idx = len(paranames)
    bound_index = len(res.bound) - 1
    fig, axes = plt.subplots(2, 3, figsize=(27, 18))
    
    for i, a in enumerate(axes.flatten()):
        it = int((i+1)*res.niter/8.)
        temp = plotting.boundplot(res, dims=(0, 1), it=it, show_live=True, fig=(fig, a), labels=paranames)
        a.set_title('Iteration {0}'.format(it), fontsize=24)
        
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{kwargs['outdir']}/boundplot_{kwargs['filename']}.png", dpi=300)
    plt.close()
    print(f"Saved bound plot: {kwargs['outdir']}/boundplot_{kwargs['filename']}.png")


if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog  [options]",
                          version="Y. Liu - 30/10/2025 - $Id$")
    parser.add_option("-d", "--datadir", type="string", dest="datadir", default=".",
                      help="Path to the data directory of the pulsar.")
    parser.add_option("-o", "--outdir", type="string", dest="outdir", default=".",
                      help="Path to the output directory.")
    parser.add_option("-p", "--psrname", type="string", dest="psrname", default="",
                      help="Name of the pulsar.")
    parser.add_option("-m", "--modelname", type="string", dest="modelname", default="",
                      help="Name of the timing model.")
    parser.add_option("-b", "--burn", type="float", dest="burn", default=0.3,
                      help="Fraction of chains to be burned.")
    parser.add_option("-e", "--evidence", action="store_true", dest="evidence", default=False,
                      help="If True: Calculate and output the final Bayesian evidence;"
                           "If False: Calculate and output the bin selection evidence.")

    (options, args) = parser.parse_args()
    args_keys = ['datadir', 'outdir', 'psrname', 'modelname', 'burn', 'evidence']
    kw_args = {key: getattr(options, key) for key in args_keys if hasattr(options, key)}
    kw_args['filename'] = kw_args['psrname']+"_"+kw_args['modelname']
    kw_args['model'] = None
    kw_args['RN'], kw_args['DM'], kw_args['SV'], kw_args['SW'] = False, False, False, False
    kw_args['rnbin'], kw_args['rnc'], kw_args['dmbin'], kw_args['dmc'] = None, None, None, None
    kw_args['svbin'], kw_args['svc'], kw_args['swbin'], kw_args['swc'] = None, None, None, None
    kw_args = get_bin_num(**kw_args)

    pklf, paramf, priorf = copy_files(**kw_args)
    res = load_dynesty_results(pklf)
    param_names = np.loadtxt(paramf, dtype=str)
    num_params = res.samples.shape[1]
    if len(param_names) != num_params:
        print(f"Warning: No. of paramters mismatch (pars.txt: {len(param_names)}, Sampling: {num_params})")
        param_names = [f"param_{i}" for i in range(num_params)]

    write_summary(res, **kw_args)
    get_dynesty_posteriors(res, param_names, **kw_args)
    extract_evidence(res, **kw_args)
    plot_corner(res, param_names, **kw_args)
    # plot_run(res, param_names, **kw_args)
    # plot_trace(res, param_names, **kw_args)
    # plot_bound(res, param_names, **kw_args)
    # plot_parameter_traces(res, param_names, **kw_args)
    # plot_parameter_histograms(res, param_names, **kw_args)
