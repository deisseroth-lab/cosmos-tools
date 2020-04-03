#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Run CNMF-E recursively on all of detected data files.
This is what we use to analyze 2p data (i.e. visual stimulus
recordings).
Created on Nov 1 2017

@author: tamachado
"""
import os
import time
import argparse
from subprocess import call

import tifffile
import numpy as np
from skimage.transform import resize

from caiman.motion_correction import MotionCorrect


def load_raw_data(input_folder, scale=1, n_frame=1000, offset=142, sz=512):
    """ Read in a folder of single tif images as a stack. """
    # Get list of tif files
    tif_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.tif'):
            tif_files.append(file)
    tif_files.sort()

    if len(tif_files) == 1:
        with tifffile.TiffFile(os.path.join(input_folder,
                               tif_files[0])) as tif:
            stack = tif.asarray()
    else:
        # Read them into a np array (requires holding whole stack in memory!)
        whole_stack = []
        offset = int(offset*scale)
        for idx, tiff_in in enumerate(tif_files):
            if idx % n_frame == 0:
                fraction = (idx+1)/len(tif_files)
                print('Reading frame ', idx, ' fraction = ', fraction)
            with tifffile.TiffFile(input_folder + os.sep + tiff_in) as im:
                im = im.asarray()
                im = resize(im, [int(im.shape[0]*scale),
                                 int(im.shape[1]*scale)], mode='constant')
                im = im[:, offset:int(np.shape(im)[1]-offset)]
                whole_stack.append(im)
        stack = np.reshape(whole_stack, [len(whole_stack),
                                         int(sz*scale), int(sz*scale)])
    return stack


def convert_and_crop_stack(input_folder, output_folder,
                           scale=0.5, name='movie.tif'):

    # Check if output files exist
    img_path = output_folder + os.sep + name
    if os.path.exists(img_path):
        raise FileExistsError('Output already exists! Stopping.')

    # Check if tif files exist in input folder
    found_tif = False
    for file_ in os.listdir(input_folder):
        if 'tif' in file_:
            found_tif = True
    if not found_tif:
        raise FileNotFoundError('No tif images found in input! Stopping.')

    # Load up the raw data from individual .tif files
    stack = load_raw_data(input_folder, scale)

    # Save out the tiff stack to a file
    stack_depth = np.shape(stack)[0]
    out_path = output_folder + os.sep + name
    print('Saving ' + out_path)
    t0 = time.time()
    stack_to_save = []
    with tifffile.TiffWriter(out_path) as tif:
        stack = np.array(stack, dtype='float32')
        for idx in range(stack_depth):
            tif.save(stack[idx, :, :])
    print('Saved. Took ', time.time() - t0, ' seconds.')
    return out_path


def preprocess_data(base, image_folder='z0',
                    scale=0.5, name='movie.tif', edge=5,
                    corrected_name='movie_corrected.tif'):
    for root, dirs, files in os.walk(base):
        offset = -1 * len(image_folder)
        if image_folder == root[offset:]:
            print('Preprocessing ', root)
            fname = root
            oname = root[:(-1-len(image_folder))]

            # Convert and crop stack
            try:
                saved_path = convert_and_crop_stack(fname, oname, scale, name)
            except (FileExistsError, FileNotFoundError) as e:
                print(oname, ' cannot be preprocessed, skipping...')
                continue

            # Motion correction, cut off edges
            mc = MotionCorrect(saved_path, 0)
            mc.motion_correct_rigid()
            with tifffile.TiffWriter(oname + os.sep + corrected_name,
                                     imagej=True) as tif:
                shifted = mc.apply_shifts_movie(saved_path)
                shifted = shifted[:, edge:-edge, edge:-edge]

                converted_stack = np.array(shifted*(2**16), dtype='uint16')
                tif.save(converted_stack)


def get_paths(base_path, valid_name, bad_name='source_extraction'):
    input_folders = []
    output_folders = []
    for root, dirs, files in os.walk(base_path):

        # Check if output already exists
        if any(bad_name in dir for dir in dirs):
            print('>>>> at ', root)
            raise IOError('Intermediate results already exist!! Stopping.')

        # Only process appropriately named files
        if valid_name in files:
                input_folders.append(root + os.sep + valid_name)
    input_folders.sort()
    for root in input_folders:
        offset = len(root.split(os.sep)[-1])
        output_folders.append(root[:-offset])

    return input_folders, output_folders


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('--matlab',
                        default='~/Matlab/bin/matlab -softwareopengl')
    parser.add_argument('--cnmf',
                        default='batch_cnmfe_2P')
    parser.add_argument('--fixer',
                        default='fix_mat_files')
    parser.add_argument('--scale',
                        default=0.5)
    args = parser.parse_args()

    raw_name = 'movie.tif'
    motion_corrected_name = 'movie_corrected.tif'

    # Load up raw tif stacks, preprocess and motion correct them
    if os.path.exists(args.base):
        args.base = os.path.abspath(args.base)
    else:
        raise IOError('Path to stack or output is invalid!')

    preprocess_data(args.base, scale=args.scale, name=raw_name,
                    corrected_name=motion_corrected_name)
    input_folders, output_folders = get_paths(args.base, motion_corrected_name)

    for idx, (inp, out) in enumerate(zip(input_folders, output_folders)):
        print('Processing ', idx)
        print('in:  ', inp)
        print('out: ', out)

        # Start MATLAB
        call(args.matlab + ' -nodesktop -nosplash -r \"' + args.cnmf +
             '(\'' + inp + '\'); exit;\"', shell=True)
    
    # Reformat all output files
    print('Fixing MAT files...')
    call(args.matlab + ' -nodesktop -nosplash -r \"' + args.fixer +
         '(\'' + args.base + '\'); exit;\"', shell=True)

    print('All files analyzed.')

