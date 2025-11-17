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


def make_json(parfile, outdir):
    """Comment out the JUMP and TN parameters in the posttn parfiles."""
    if "-posttn" in parfile:
        psrn = parfile.split("/")[-1].split("-posttn")[0]
        parlines = ["{\n"]
        with open(parfile) as f:
            for line in f:
                e = line.split()
                if e[0] == "TNEF" and len(e) > 2:
                    newline = '    "' + psrn + '_' + e[-2] + '_efac": ' + e[-1] + ',\n'
                    parlines.append(newline)
                elif e[0] == "TNEQ" and len(e) > 2:
                    newline = '    "' + psrn + '_' + e[-2] + '_log10_tnequad": ' + e[-1] + ',\n'
                    parlines.append(newline)
                elif e[0] == "TNECORR" and len(e) > 2:
                    tnecorr = np.log10(float(e[-1]))-6
                    newline = '    "' + psrn + '_' + e[-2] + '_log10_ecorr": ' + str(tnecorr) + ',\n'
                    parlines.append(newline)
        parlines[-1] = parlines[-1].rsplit(",",1)[0]
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
