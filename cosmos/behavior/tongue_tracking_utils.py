#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Helper functions for segmenting tongues/processing behavior data.

Created on Nov 16 2017

@author: tamachado
"""
from scipy.ndimage.measurements import center_of_mass
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.io as sio
import cv2


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from stack exchange
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dicts
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def plot_peri_spout_density(trial, frame_idx, tongue_positions,
                            spout_positions,
                            trial_assignments, min_area=100):
    columns = 7

    # Plot the desired trial only
    rng = 40
    tongue_centers = []
    chosen_bouts = np.where(trial_assignments == trial)[0]
    rows = int(len(chosen_bouts)/columns)+1
    plt.figure(figsize=(20, rows*3.25))
    ct = 1
    print('TRIAL', trial, 'NOT PLOTTING DENSITIES LESS THAN ', min_area)
    for idx in range(len(frame_idx)):
        if idx not in chosen_bouts:
            continue

        # Superimpose all values from tongue masks
        density = np.zeros(((rng*2)+1, (rng*2)+1))
        area = []
        for ff in frame_idx[idx][1:]:
            mask_sz = np.shape(tongue_positions[ff]['mask'])
            ry = slice(np.max([spout_positions[ff, 1]-rng, 0]),
                       np.min([spout_positions[ff, 1]+rng+1, mask_sz[1]]))
            rx = slice(np.min([spout_positions[ff, 0]+rng+1, mask_sz[0]]),
                       np.max([spout_positions[ff, 0]-rng, 0]), -1)
            try:
                density += tongue_positions[ff]['mask'][rx, ry]
            except Exception:
                print('skipped frame...')
                continue
            area.append(tongue_positions[ff]['area'])
        if np.median(area) < min_area:
            continue

        plt.subplot(rows, columns, ct)
        ct += 1
        centroid = center_of_mass(density)
        tongue_centers.append(centroid)
        plt.imshow(1-density, cmap='bone')
        plt.plot(centroid[1], centroid[0], 'r.')

        # Plot the spout positions
        plt.plot(rng, rng, 'r.')
        plt.axis('off')

        # How many frames long was this bout?
        plt.title('trial ' + str(int(trial_assignments[idx])) + '\n' +
                  'spout position = ' + str([spout_positions[ff, 1],
                                            spout_positions[ff, 0]]) + '\n' +
                  'start time (s) = ' + str(frame_idx[idx][0]/200))

    # Get rid of whitespace
    plt.subplots_adjust(wspace=0, hspace=0.4)

    # Return the centroid of each tongue tensity and the spout position
    return tongue_centers, [rng, rng]


def plot_tongue_densities(trial, stack, frame_idx, tongue_positions,
                          target_region, traj_tongues,
                          traj_spouts, trial_assignments,
                          show_trajectory=False, min_area=100):
    columns = 7

    # Plot the desired trial only
    chosen_bouts = np.where(trial_assignments == trial)[0]
    rows = int(len(chosen_bouts)/columns)+1
    plt.figure(figsize=(20, rows*3.25))
    ct = 1
    print('TRIAL', trial, 'NOT PLOTTING DENSITIES LESS THAN ', min_area)
    for idx, (trajectory, spout) in enumerate(zip(traj_tongues, traj_spouts)):
        if idx not in chosen_bouts:
            continue

        # Superimpose all values from tongue masks
        x = tongue_positions[frame_idx[idx][0]]['mask']
        x = x - np.min(x)
        area = []
        for ff in frame_idx[idx][1:]:
            x += tongue_positions[ff]['mask']
            area.append(tongue_positions[ff]['area'])
        if np.median(area) < min_area:
            continue

        plt.subplot(rows, columns, ct)
        ct += 1

        # Plot background image (first image from bout) and density
        plt.imshow(stack[frame_idx[idx][0], :, :, :])
        plt.imshow(1-x, alpha=.6, cmap='bone')

        # Zoom into area around mouth
        plt.xlim([target_region[1].start, target_region[1].stop])
        plt.ylim([target_region[0].start, target_region[0].stop])

        # Plot trajectory with start point in green and endpoint in red
        if show_trajectory:
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'w', lw=2)
            plt.plot(trajectory[0, 0], trajectory[0, 1], 'g.', markersize=15)
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'r.', markersize=15)

        # Plot the spout positions
        plt.plot(spout[:, 1], spout[:, 0], 'k.')
        plt.axis('off')

        # How many frames long was this bout?
        plt.title('trial ' + str(int(trial_assignments[idx])) + '\n' +
                  'start frame = ' + str(frame_idx[idx][0]) + '\n' +
                  str(len(trajectory)) + ' frames (' +
                  str(round(len(trajectory)/200, 3)) + 's)')

    # Get rid of whitespace
    plt.subplots_adjust(wspace=0, hspace=0.4)


def plot_tongue_tracking(stack, frames, tongue_positions,
                         spout_positions, n_spout=1,
                         min_area=100, interval=50):
    """ Plot an animation of the segmentation. """

    def init():
        """ Setup painters to update on each frame. """
        frame.set_data(stack[0, :, :, :])
        if n_spout == 1:
            spout.set_data([], [])
        tongue_centroid.set_data([], [])
        tongue_contour.set_data([], [])
        return (frame, spout, tongue_centroid, tongue_contour)

    def animate(idx):
        """ Update painters on each frame. """
        frame.set_data(stack[idx, :, :, :])
        if n_spout == 1:
            spout.set_data(spout_positions[idx, 1], spout_positions[idx, 0])

        if tongue_positions[idx]['area'] > min_area:
            tongue_centroid.set_data(tongue_positions[idx]['centroid'][0],
                                     tongue_positions[idx]['centroid'][1])
            tongue_contour.set_data(tongue_positions[idx]['contour'][:, 0],
                                    tongue_positions[idx]['contour'][:, 1])
        else:
            tongue_centroid.set_data([], [])
            tongue_contour.set_data([], [])
        return (frame, spout, tongue_centroid, tongue_contour)

    # Setup the figure layout
    fig = plt.figure(figsize=(10, 10))
    frame = plt.imshow(stack[0, :, :, :])
    spout, = plt.plot(0, 0, 'r.', markersize=30)
    tongue_centroid, = plt.plot(0, 0, 'b.', markersize=30)
    tongue_contour, = plt.plot(0, 0, 'b', lw=2)
    plt.axis('off')
    plt.close()

    # Update the figure on each frame
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval,
                                   blit=True)

    return anim


def get_tongue_position(frame, min_color, max_color, target_region=None):
    """Extract pixels corresponding to tongue from a given RGB frame."""

    # If specified, zero out pixels not in target region
    if target_region:
        zero_mask = np.zeros(frame.shape, np.uint8)
        zero_mask[target_region[0], target_region[1], :] = 255
        frame = cv2.bitwise_and(frame, zero_mask)

    # Gaussian blur the image
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

    # Find spout-colored pixels
    mask = cv2.inRange(hsv, min_color, max_color)

    # Get the 8-connected components
    count, comps = cv2.connectedComponents(mask, connectivity=8)

    # Get the second largest component
    sizes = [np.sum(np.where(comps == val)) for val in range(count)]
    if len(sizes) < 2:
        empty_tongue = {}
        empty_tongue['centroid'] = [np.nan, np.nan]
        empty_tongue['area'] = np.nan
        return empty_tongue
    biggest = np.argsort(sizes)[-2]
    comp = np.zeros(np.shape(comps))
    comp[np.where(comps == biggest)] = 1

    # Get a contour around it
    _, contour, _ = cv2.findContours(np.array(comp, dtype=np.uint8),
                                     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0]
    area = cv2.contourArea(contour)

    # Get centroid of the contour
    if area > 10:
        M = cv2.moments(contour)
        centroid = [M['m10']/M['m00'], M['m01']/M['m00']]
    else:
        centroid = [np.nan, np.nan]

    # Return the contour around the tongue as well as its centroid
    tongue = {}
    tongue['contour'] = np.squeeze(contour)
    if len(tongue['contour']) > 2:
        tongue['contour'] = np.vstack((tongue['contour'],
                                      tongue['contour'][0, :]))
    tongue['area'] = area
    tongue['centroid'] = centroid
    tongue['mask'] = comp
    return tongue


def get_spout_position(frame, template, target_region=None, n_spout=1):
    """Extract centroid corresponding to waterspout from a given RGB frame."""

    # If specified, zero out pixels not in target region
    if target_region:
        zero_mask = np.zeros(frame.shape, np.uint8)
        zero_mask[target_region[0], target_region[1], :] = 255
        frame = cv2.bitwise_and(frame, zero_mask)

    # Convert frame and template to HSV
    roi_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find spout-colored pixels
    filtered = cv2.matchTemplate(frame_hsv, roi_hsv, cv2.TM_CCOEFF)

    # Return the coordinate of the max point(s)
    if n_spout > 1:
        max_arg = np.argsort(np.ravel(filtered))
        return np.array(np.unravel_index(max_arg[-1:(-1*n_spout-1):-1],
                        np.shape(filtered))).T
    else:
        return np.unravel_index(np.argmax(filtered), np.shape(filtered))


def get_spout_position_color(frame, min_color, max_color,
                             target_region=None, return_mask=False):
    """Extract centroid corresponding to waterspout from a given RGB frame."""

    # If specified, zero out pixels not in target region
    if target_region:
        zero_mask = np.zeros(frame.shape, np.uint8)
        zero_mask[target_region[0], target_region[1], :] = 255
        frame = cv2.bitwise_and(frame, zero_mask)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find spout-colored pixels
    mask = cv2.inRange(hsv, min_color, max_color)

    # Return the raw mask or the coordinate of the max point
    if return_mask:
        return mask
    else:
        max_ind = np.argmax(mask*np.sum(hsv, 2))
        return np.unravel_index(max_ind, mask.shape)
