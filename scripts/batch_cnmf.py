#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Stripped down script for just calling Matlab CNMFe
recursively on all detected data files within a given
directory.
This is NOT the primary script we use for processing
COSMOS data.

Created on Nov 1 2017

@author: tamachado
"""
import os
import argparse
from subprocess import call


def get_paths(base_path, valid_names, mouse, bad_name='source_extraction'):
    input_folders = []
    output_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(bad_name in dir for dir in dirs):
            print('>>>> at ', root)
            raise IOError('Intermediate results already exist!! Stopping.')
        for name in valid_names:
            if mouse in root and name in files:
                input_folders.append(root + os.sep + name)
    input_folders.sort()
    for root in input_folders:
        offset = len(root.split(os.sep)[-1])
        output_folders.append(root[:-offset])

    return input_folders, output_folders


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    base = '/media/deisseroth/Data/Analysis/COSMOS/20171016'
    parser.add_argument('--mouse',
                        default='cux2ai148m1')
    parser.add_argument('--base',
                        default=base)
    parser.add_argument('--matlab',
                        default='~/Matlab/bin/matlab -softwareopengl')
    parser.add_argument('--cnmf',
                        default='COSMOS_analysis')
    args = parser.parse_args()

    # Names of valid files to process
    names = ['vid.tif']

    if os.path.exists(args.base):
        args.base = os.path.abspath(args.base)
    else:
        raise IOError('Path to stack or output is invalid!')

    input_folders, output_folders = get_paths(args.base,
                                              names,
                                              args.mouse)
    for idx, (inp, out) in enumerate(zip(input_folders, output_folders)):
        print('Processing ', idx)
        print('in:  ', inp)
        print('out: ', out)

        # Start MATLAB
        call(args.matlab + ' -nodesktop -nosplash -r \"' + args.cnmf +
             '(\'' + inp + '\'); exit;\"', shell=True)
