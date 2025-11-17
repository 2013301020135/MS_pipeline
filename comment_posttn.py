#!/usr/bin/env python
"""Tidy up posttn par files by commenting out all JUMP and TN parameters. Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Tidy up posttn par files by commenting out all JUMP and TN parameters.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+',
                    help='Parameter files for pulsars used in simulation')
parser.add_argument('-d', '--datadir', type=str, default=None, help='Path to the directory containing the par files')


def comment_posttn(parfile):
    """Comment out the TN parameters in the posttn parfiles."""
    if "-posttn" in parfile:
        parlines = []
        with open(parfile) as f:
            for line in f:
                e = line.split()
                if "TN" in e[0]:
                    parlines.append("# "+line)
                else:
                    parlines.append(line)
        newpar = parfile.split("-posttn")[0]+"_all.par"
        with open(newpar, "w") as newf:
            newf.writelines(parlines)
        return newpar
    else:
        return parfile


args = parser.parse_args()
if args.datadir is not None:
    posttn_files = sorted(glob.glob(os.path.join(args.datadir, "*.par")))
    par_files = sorted(comment_posttn(pfs) for pfs in posttn_files)
else:
    par_files = sorted(comment_posttn(pfs) for pfs in args.parfile)
