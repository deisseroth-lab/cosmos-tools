"""
A module with helper functions for evaluating, culling, and
merging cells extracted with CNMF.

Created on March 22 2018

@author: ikauvar@stanford.edu
"""
from past.utils import old_div
from warnings import warn
import warnings
import time

import numpy as np
import skimage.draw
import matplotlib as mpl
from skimage import measure
import scipy.ndimage.measurements as measurements
import scipy.sparse as sparse
from skimage.transform import rotate
from scipy.stats import pearsonr
from scipy.ndimage.interpolation import rotate as rot

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


from IPython.core.debugger import set_trace


def get_cm(footprints):
    """
    Get average size and centroid of each ROI.
    :param footprints: a 3D np.array [X x Y x nROIs]
    :returns size_map: a 2D np.array
    """
    all_cm = np.zeros((footprints.shape[2], 2))
    all_area = np.zeros((footprints.shape[2], 1))
    for i in range(footprints.shape[2]):
        f = footprints[:, :, i]
        cm = measurements.center_of_mass(f)
        c = get_contour_fast(f)

        # returns coordinates to fill polygon
        if len(c) > 0:
            rr, cc = skimage.draw.polygon(c[:, 0], c[:, 1])
            all_cm[i, :] = cm[:]
            all_area[i] = len(rr)
        else:
            all_cm[i, :] = np.zeros(np.shape(cm))
            all_area[i] = 0
    return all_cm, all_area


def get_contour_fast(cell):
    """
    Returns the largest cell ROI contour in the
    provided image. Output can be plotted with bokeh using:
        p.patches(xs=[contour[:,0]], ys=[contour[:,1]],
                  color=["firebrick"], fill_alpha=[0.0],
                  line_alpha = [1.0], line_width=2)
    :param cell: an image containing the ROI of a cell.
    :return contour: (2d ndarray)
    """
    contours = measure.find_contours(cell, 0.1*np.amax(cell))
    if len(contours) > 0:
        contour = contours[np.argmax([len(x) for x in contours])]
        contour = np.fliplr(contour)
    else:
        contour = []
    return contour


def get_cell_zoom(img, npixels=100):
    """
    Returns a cropped zoom-in of an image of a cell ROI.
    :param cell: an image containing the ROI of a cell.
    :param npixels: side length of the square cropped region.
    :return: cropped_cell
    """
    npixels = min(npixels, min(np.shape(img)))
    max_col = np.argmax(np.amax(img, axis=0))
    max_row = np.argmax(np.amax(img, axis=1))

    start_col = int(max(0, max_col - (npixels / 2)))
    start_row = int(max(0, max_row - (npixels / 2)))
    end_col = int(min(img.shape[1], start_col + npixels))
    end_row = int(min(img.shape[0], start_row + npixels))

    zoom_img = np.zeros((npixels, npixels))
    c0 = int(start_col - max(0, (max_col - (npixels / 2))))
    r0 = int(start_row - max(0, (max_row - (npixels / 2))))
    zoom_img[r0:r0 + end_row - start_row,
             c0:c0 + end_col - start_col] = img[start_row:end_row,
                                                start_col:end_col]

    return zoom_img


def discrete_radon(img, steps):
    """
    Computes radon transform (the projection through
    an image at different angles with an angular
    step size of 180degrees/steps.
    :param img: ndarray.
    :param steps: number of angular steps. More steps takes longer.
    :return: The discrete radon transform.
    """
    R = np.zeros((steps, len(img)), dtype='float64')
    for s in range(steps):
        rotation = rotate(img, -s * 180 / steps).astype('float64')
        R[s, :] = np.sum(rotation, axis=1)
    return R


def get_aspect_ratio(img):
    """
        Compute the aspect ratio of the object an image.
        For example, the image may have the spatial fooprint of a neuron,
        and we want to ensure that only neurons with roughly symmetrical
        footprints are accepted.
        Rather than fitting a gaussian, this function computes the radon
        transform, which is the projection of the image across different
        angles. Then the aspect ratio is computed based on the maximum and
        minimum width projection. This accounts for the possibility that the
        primary and secondary axes are tilted relative to vertical.
        :params img: an image (with, presumably, a single object in it)
        :returns aspect_ratio: the ratio between the maximum and minimum widths
                     of the object (potentially along a tilted set of axes).
     """

    r = discrete_radon(img, 20)
    # Assumes images are scaled such that 'non-zero'
    # signals have values greater than at least 1.
    R = r > 1
    Rwidths = np.sum(R, axis=1)
    aspect_ratio = (np.amax(Rwidths) / np.amin(
        Rwidths[np.where(Rwidths > 0)[0]]))
    return aspect_ratio


def classify_neurons(neuron_struct, corr_thresh=0.8, aspect_ratio_thresh=2):
    """
        Automatically classify traces as being neuron-like or not (i.e. blood
        vessels).

        :param neuron_struct: struct containing results from CNMF-E. Must
                              contain the fields 'C', 'C_raw', and 'A'
        :param corr_thresh: threshold for correlation between the deconvolve
                            and the raw trace. (Technically, the correlation
                            between parts of the signal that are multiple
                            standard deviations above the baseline).
        :param aspect_ratio_thresh: threshold on acceptalbe aspect-ratio of the
                                    spatial footprint
                                    (which should be fairly symmetric).

        :returns keep_cells: np.array of length ncells, where 1 indicates a
                             good neuron and 0 indicates a bad neuron.
    """
    do_plot = False

    C = neuron_struct['C']
    C_raw = neuron_struct['C_raw']
    A = neuron_struct['A']
    ncells = C.shape[0]
    base_im = neuron_struct['Cn']

    if len(A.shape) < 3:
        footprints = np.array(neuron_struct['A'][:, :].todense())
        footprints = np.reshape(footprints,
                                (base_im.shape[0], base_im.shape[1],
                                 ncells), order='F')
    else:
        footprints = A

    keep_cells = -1 * np.ones((ncells, 1))
    corrs = np.zeros((ncells, 1))
    aspect_ratios = np.zeros((ncells, 1))

    if do_plot:
        plt.figure()
    for i in range(ncells):
        # Look at correlation only during large events.
        peak_thresh = 2 * np.std(C[i, :])
        inds = np.where(C[i, :] > peak_thresh)
        if len(inds) > 0:
            try:
                cc, _ = pearsonr(np.squeeze(C[i, inds]),
                                 np.squeeze(C_raw[i, inds]))
            except:
                cc = 0
            corrs[i] = cc

            thresh = aspect_ratio_thresh
            if thresh is not None:
                a = footprints[:, :, i]
                aspect_ratio = get_aspect_ratio(a)
                aspect_ratios[i] = aspect_ratio

            if corr_thresh is not None and thresh is not None:
                keep_cells[i] = (cc > corr_thresh and aspect_ratio < thresh)
            elif corr_thresh is not None and thresh is None:
                keep_cells[i] = cc > corr_thresh
            elif corr_thresh is None and thresh is not None:
                keep_cells[i] = aspect_ratio < thresh

                if do_plot:
                    plt.subplot(121)
                    plt.plot(C[i, :])
                    plt.plot(C_raw[i, :])
                    plt.title(str(i) + ': ' + str(cc))
                    plt.subplot(122), plt.imshow(a)
                    plt.title(str(aspect_ratio))
                    # time.sleep(1)
                    plt.show()
            if corr_thresh is not None and aspect_ratio_thresh is not None:
                print(str(aspect_ratio) + ' ' + str(cc))
            elif corr_thresh is not None and aspect_ratio_thresh is None:
                print(str(cc))
            elif corr_thresh is None and aspect_ratio_thresh is not None:
                print(str(aspect_ratio))

    return keep_cells, corrs, aspect_ratios


def fit_polynomial(xx, yy, xx_plot, order):
    """
    Fits a polynomial of provided 'order'
    to the data yy as a function of xx.
    :param xx: an np.array
    :param yy: an np.array
    :param xx_plot: x values
    :param order: order of the polynomial. i.e. 2 or 3
    :returns yy_plot: values evaluated at xx_plot
    :returns model: the fit model
    """

    X = xx[:, np.newaxis]

    model = make_pipeline(PolynomialFeatures(2), Ridge())
    model.fit(X, yy)
    yy_plot = model.predict(X)

    return yy_plot, model


def fit_transform(c1, c2, transform='euclidean'):
    """
    Given two lists of paired coordinates, c1, and c2
    Fit a transform that transforms points c2 to align
    with c1.
    :param transform: 'euclidean' (rigid), 'affine' (with scaling)
    """
    return skimage.transform.estimate_transform(transform, c2, c1)


def align_image(tform, img2, output_shape=None):
    return skimage.transform.warp(img2, tform.inverse,
                                  output_shape=output_shape)


def align_stacks(tform, stack2):
    """
    Given a stack [Y x X x T],
    aligns each timepoint according
    to the tform (which is the output of fit_transform).
    """
    aligned_stack = np.zeros_like(stack2)
    for t in range(stack2.shape[2]):
        aligned_stack[:, :, t] = align_image(tform,
                                             np.squeeze(stack2[:, :, t]))

    return aligned_stack


def crop_images(img1, img2):
    """
    Crops images that have been transformed
    according to manually selected keypoints
    so that they are the same dimensions.
    It turns out that the indexing is such that
    grabbing the largest indices lines up the images.
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    H = min(h1, h2)
    W = min(w1, w2)
    return [img1[:H, :W], img2[:H, :W]]


def crop_stacks(stack1, stack2):
    """
    Crops stacks [X x Y x T] that have been transformed
    according to manually selected keypoints
    so that they are the same dimensions.
    It turns out that the indexing is such that
    grabbing the largest indices lines up the images.
    """
    h1, w1, t1 = stack1.shape
    h2, w2, t2 = stack2.shape
    H = min(h1, h2)
    W = min(w1, w2)
    return [stack1[:H, :W, :], stack2[:H, :W, :]]


def transform_points(tform, xy_coords):
    return skimage.transform.matrix_transform(xy_coords, tform.params)


def get_roi_map(all_cm, all_area, imshape):
    """
    Return an image where at each coordinate
    in all_cm, the value is the corresponding
    all_area.

    """
    NX = imshape[0]
    NY = imshape[1]
    r_map = np.zeros((NX, NY))
    out_of_bounds = np.where(np.logical_or(all_cm[:, 0] > NX,
                             all_cm[:, 1] > NY))[0]
    if len(out_of_bounds) > 0:
        warnings.warn('There are '+str(len(out_of_bounds))+' rois out of bounds in the aligned and cropped images.'
                      'This is likely because you cropped one of the bot_focus or top_focus images much smaller than'
                      'the other one, and therefore padding got cut off. This error arises in get_roi_map. You can'
                      'either recrop and reimport data, or just let it slide, depending on the application.')
        all_cm = np.delete(all_cm, out_of_bounds, axis=0)
        all_area = np.delete(all_area, out_of_bounds, axis=0)
    r_map[all_cm[:, 0].astype(int), all_cm[:, 1].astype(int)] = all_area[:, 0]

    return r_map


def com(A, d1, d2):
    """Calculation of the center of mass for spatial components

       From Caiman: https://github.com/flatironinstitute/CaImAn
       @author: agiovann

     Inputs:
     ------
     A:   np.ndarray
          matrix of spatial components (d x K)

     d1:  int
          number of pixels in x-direction

     d2:  int
          number of pixels in y-direction

     Output:
     -------
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    nr = np.shape(A)[-1]
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(list(range(d1)),
                        axis=1))
    Coor['y'] = np.kron(np.expand_dims(list(range(d2)), axis=1),
                        np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    cm[:, 0] = old_div(np.dot(Coor['x'].T, A), A.sum(axis=0))
    cm[:, 1] = old_div(np.dot(Coor['y'].T, A), A.sum(axis=0))

    return cm


def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9,
                  display_numbers=True, display_reduced_index=True, max_number=None,
                  cmap=None, swap_dim=False, colors=(1, 1, 1, 1), vmin=None,
                  vmax=None, highlight_neurons=None,
                  highlight_color=(1, 1, 1, 1),
                  just_show_highlighted=False,
                  contour_linewidth=1, rotate_vertical=False,
                  show_footprints=False,
                  **kwargs):
    #TODO: Refactor this in terms of get_contours_fast and get_cm
    """Plots contour of spatial components against a background image
       and returns their coordinates

       Modified from Caiman: https://github.com/flatironinstitute/CaImAn
       @author: agiovann, ikauvar

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding:
                  'max' sets to zero pixels that have value less
                  than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a
                  specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility.
               If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display the id number of each ROI if checked (default True)

     display_reduced_index: Boolean. Only applies if display_number is true.
                Instead of displaying the number that is the original cell_id,
                display the number that is the index into just the cells
                that are highlighted.

     max_number:    int
               Display the number for only the first max_number components
               (default None, display all numbers)

     cmap:     string
               User specifies the colormap for displaying the background image
               (default None, default colormap)

     highlight_neurons:     np.ndarray or list
                           If not None, will highlight and number
                           the specified subset of neuron

     highlight_color: For a single color, provide an RGBA tuple (i.e. (1,1,1,1)
                      For a colormap, provide a colormap (i.e. plt.cm.hsv)

     show_footprints: bool. If True then display the
                            footprint corresponding to each cell
                            in addition to the contour of that footprint.

     Returns:
     --------
     Coor: list of coordinates with center of mass,
        contour plot coordinates and bounding box for each component
    """
    if sparse.issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    if max_number is None:
        max_number = nr

    if thr is not None:
        thr_method = 'nrg'
        nrgthr = thr
        warn("The way to call utilities.plot_contours has changed.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    ax = plt.gca()
    if vmax is None and vmin is None:
        ax.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                  vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        ax.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)

    coordinates = []
    cm = com(A, d1, d2)

    polygons = []

    cell_list = range(np.minimum(nr, max_number))
    tr = mpl.transforms.Affine2D().rotate_deg(-90)
    for i in cell_list:
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = np.squeeze(mean_frames[0, :, :])

        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)+1e-10
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')

        if rotate_vertical:
            Bmat = np.flipud(Bmat)
        contours = measure.find_contours(Bmat, thr)

        if len(contours) > 0:
            contour = contours[np.argmax([len(x) for x in contours])]
            if rotate_vertical:
                pol = Polygon(contour)
            else:
                pol = Polygon(np.fliplr(contour))
            polygons.append(pol)

    p = PatchCollection(polygons, edgecolors=(colors,),
                        linewidths=(contour_linewidth,),
                        facecolors=((0, 0, 0, 0),))

    ax.add_collection(p)

    if highlight_neurons is not None:
            tt = type(highlight_color)
            if tt == tuple:
                facecolors = (highlight_color,)
            elif (tt == np.ndarray or tt == list):
                # Assume we provided a unique color for each point
                facecolors = highlight_color
            else:
                # Assumes a colormap is provided
                facecolors = [highlight_color(x/float(len(highlight_neurons)))
                              for x in np.arange(len(highlight_neurons))]
            if just_show_highlighted:
                p = PatchCollection(polygons, edgecolors=facecolors,
                                    linewidths=(contour_linewidth,),
                                    facecolors=facecolors)
                ax.add_collection(p)
            else:
                highlight_polygons = [polygons[x] for x in highlight_neurons]
                ph = PatchCollection(highlight_polygons, edgecolors=facecolors,
                                     linewidths=(contour_linewidth,),
                                     facecolors=facecolors)
                ax.add_collection(ph)

    if display_numbers:
        if highlight_neurons is None:
            numbers_indices = range(np.minimum(nr, max_number))
            numbers_labels = numbers_indices
        else:
            if just_show_highlighted:
                numbers_indices = range(np.minimum(nr, max_number))
                if display_reduced_index:
                    numbers_labels = numbers_indices
                else:
                    numbers_labels = highlight_neurons
            else:
                numbers_indices = highlight_neurons
                if display_reduced_index:
                    numbers_labels = np.arange(len(numbers_indices))
                else:
                    numbers_labels = numbers_indices
        for ind, i in enumerate(numbers_indices):
            label = numbers_labels[ind]
            if swap_dim:
                ax.text(cm[i, 0]+3, cm[i, 1]-3, str(label),
                        color=(0, 1, 0, 1), weight='bold')
            else:
                # ax.text(cm[i, 1], cm[i, 0], str(label),
                #         color=(0, 1, 0, 1), weight='bold') ### *** THIS WAS PRE-9/20/18. THE CHANGE TO THIS MAY HAVE SCREWED THINGS UP. BE CAREFUL.
                ax.text(d1-cm[i, 0]+3, cm[i, 1]-3, str(label),
                        color=(0, 1, 0, 1), weight='bold')

    coordinates = {'cm': cm, 'patches': polygons}

    return coordinates

# def save_image_no_border(data, filename):
#     sizes = np.shape(data)
#     fig = plt.figure()
#     fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(data)
#     plt.savefig(filename, dpi=sizes[0], cmap='hot')
#     plt.close()