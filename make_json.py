#!/usr/bin/env python
"""Write noise files by reading TN parameters from parameter files. Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob
import numpy as np

parser = argparse.ArgumentParser(description='Write noise files by reading TN parameters from parameter files.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+',
                    help='Parameter files for pulsars used in simulation')
parser.add_argument('-d', '--datadir', type=str, default=None, help='Path to the directory containing the par files')
parser.add_argument('-o', '--outdir', type=str, default=None, help='Path to the directory for the noise files output')
parser.add_argument("-c", "--ecorr", action="store_true", dest="ecorr", default=False,
                    help="If True: Add ecorr parameters for JUMPs in par file; If False: Ignore ecorr for JUMPs.")


def make_json(parfile, outdir):
    """Comment out the JUMP and TN parameters in the posttn parfiles."""
    if "-posttn" in parfile:
        psrn = parfile.split("/")[-1].split("-posttn")[0]
        parlines = ["{\n"]
        added_names = set()
        jump_backends = []
        with open(parfile) as f:
            for line in f:
                e = line.split()
                if e[0] == "TNEF" and len(e) > 2:
                    name = f"{psrn}_{e[-2]}_efac"
                    if name not in added_names:
                        added_names.add(name)
                        parlines.append(f'    "{name}": {e[-1]},\n')
                elif e[0] == "TNEQ" and len(e) > 2:
                    name = f"{psrn}_{e[-2]}_log10_tnequad"
                    if name not in added_names:
                        added_names.add(name)
                        parlines.append(f'    "{name}": {e[-1]},\n')
                elif e[0] == "TNECORR" and len(e) > 2:
                    name = f"{psrn}_{e[-2]}_log10_ecorr"
                    if name not in added_names:
                        added_names.add(name)
                        tnecorr = np.log10(float(e[-1])) - 6
                        parlines.append(f'    "{name}": {tnecorr},\n')
                elif e[0] == "JUMP" and len(e) > 3:
                    backend = e[2]
                    jump_backends.append(backend)
        for backend in set(jump_backends):
            efac_name = f"{psrn}_{backend}_efac"
            equad_name = f"{psrn}_{backend}_log10_tnequad"
            ecorr_name = f"{psrn}_{backend}_log10_ecorr"
            if efac_name not in added_names:
                added_names.add(efac_name)
                parlines.append(f'    "{efac_name}": 1.0,\n')
            if equad_name not in added_names:
                added_names.add(equad_name)
                parlines.append(f'    "{equad_name}": -10.0,\n')
            if args.ecorr and ecorr_name not in added_names:
                added_names.add(ecorr_name)
                parlines.append(f'    "{ecorr_name}": -10.0,\n')
        parlines[-1] = parlines[-1].rsplit(",", 1)[0]
        parlines.append("\n}")
        noisefile = os.path.join(outdir, psrn+".json")
        with open(noisefile, "w") as newf:
            newf.writelines(parlines)
        return noisefile


args = parser.parse_args()
if args.datadir is not None:
    posttn_files = sorted(glob.glob(os.path.join(args.datadir, "*.par")))
    par_files = sorted(make_json(pfs, args.outdir) for pfs in posttn_files)
else:
    par_files = sorted(make_json(pfs, args.outdir) for pfs in args.parfile)
