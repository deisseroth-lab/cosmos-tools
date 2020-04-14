from collections import defaultdict
from functools import partial
import warnings
import copy
import os

from IPython.display import display
from ipywidgets import widgets
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from scipy.ndimage import label, center_of_mass
from sklearn.cluster import KMeans
import scipy.sparse as sparse
import scipy.stats as stats
from skimage import measure
from scipy import linalg

from bokeh.io import output_file, show, output_notebook, push_notebook
from bokeh.layouts import widgetbox, column, row, layout
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import Range1d
import bokeh.models as models
import bokeh.plotting as bpl
import bokeh

import tifffile

from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
from caiman.utils.visualization import plot_contours
from past.utils import old_div


class CellSorter():

    def __init__(self, results, video, background_image):

        self.vid = video
        self.results = results
        self.baseIm = background_image

        self._initialize_plot()

    def _initialize_plot(self):
        cell_id = 1
        self.ncells = self.vid.shape[2]
        cell = self.vid[:, :, cell_id]
        trace = self.results['C'][cell_id, :]
        trace_raw = self.results['C_raw'][cell_id, :]

        cell_zoom = get_cell_zoom(cell)
        cell_zoom_contour = get_contour_fast(cell_zoom)
        cell_contour = get_contour_fast(cell)

        # Initialize plots
        self.p1 = figure(plot_width=300, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        img = self.baseIm
        self.r1_img = self.p1.image(image=[img], x=[0], y=[0], 
                                    dw=[img.shape[1]], dh=[img.shape[0]])
        self.r1_contour = self.p1.patches(xs=[cell_contour[:, 0]],
                                          ys=[cell_contour[:, 1]],
                                          color=["firebrick"],
                                          fill_alpha=[0.0],
                                          line_alpha=[1.0], line_width=2)
        self.p1.x_range = Range1d(0, img.shape[0])
        self.p1.y_range = Range1d(0, img.shape[1])

        self.p3 = figure(plot_width=300, plot_height=300, 
                         toolbar_location="above", 
                         tools="pan,wheel_zoom,box_zoom,reset")
        img = cell_zoom
        self.r3_img = self.p3.image(image=[img], x=[0], y=[0],
                                    dw=[img.shape[1]], dh=[img.shape[0]])
        self.r3_contour = self.p3.patches(xs=[cell_zoom_contour[:, 0]],
                                          ys=[cell_zoom_contour[:, 1]],
                                          color=["firebrick"], fill_alpha=[0.0], 
                                          line_alpha=[1.0], line_width=2)
        self.p3.x_range = Range1d(0, img.shape[0])
        self.p3.y_range = Range1d(0, img.shape[1])

        self.p4 = figure(plot_width=300, plot_height=300, 
                         toolbar_location="above", 
                         tools="pan,wheel_zoom,box_zoom,reset")
        self.r4 = self.p4.image(image=[cell], x=[0], y=[0], dw=[1], dh=[1])
        self.p4.x_range = Range1d(0, 1)
        self.p4.y_range = Range1d(0, 1)

        self.p2 = figure(plot_width=900, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        nx = len(trace)
        self.r2_raw = self.p2.line(x=range(nx), y=trace_raw,
                                   line_width=2, color='firebrick')
        self.r2_deconv = self.p2.line(x=range(nx), y=trace,
                                      line_width=2, color='blue')
        self.p2.x_range = Range1d(0, nx)
        self.p2.y_range = Range1d(np.min(trace_raw), np.max(trace_raw))

        # Handles for updating data
        self.rs1_contour = self.r1_contour.data_source
        self.rs2_raw = self.r2_raw.data_source
        self.rs2_deconv = self.r2_deconv.data_source
        self.rs3_img = self.r3_img.data_source
        self.rs3_contour = self.r3_contour.data_source
        self.rs4 = self.r4.data_source

        # Add widgets
        self.w1 = widgets.Text(value='', placeholder='k=keep, d=delete',
                               description='[k]eep/[d]elete? ',
                               disabled=False, continuous_update=True)

        self.w2 = widgets.IntSlider(value=0, min=0, max=self.ncells, step=1,
                                    description='Frame', disabled=False,
                                    continuous_update=True, readout=True,
                                    readout_format='d')

        # Render plots and widgets
        self.handle = show(column(row(self.p1, self.p3), self.p2),
                           notebook_handle=True)
        display(self.w1, self.w2)

        # Setup widget callbacks
        self.iter = np.ones((1, 1))
        self.entries = []
        self.bad_cells = []
        self.good_cells = []

        # Link up widget callbacks
        self.w1.on_submit(self._handle_submit)
        self.w2.observe(self._handle_slider, names='value')

    def _handle_submit(self, sender):
        cell_id_old = int(self.iter[0])
        if self.w1.value == 'd':
            self.bad_cells.append(cell_id_old)
        else:
            self.good_cells.append(cell_id_old)

        self.iter[0] += 1
        self.cell_id = int(self.iter[0])
        self._update_cell()
        self.w1.value = ''  # Reset the text box to empty
        self.w2.value = self.cell_id

    def _handle_slider(self, change):
        self.cell_id = change.new
        self._update_cell()

    def _update_cell(self):
        # Note: when updating data in bokeh, need to be sure
        # to update ALL of the fields, or it may not work.
        cell_id = self.cell_id
        cell = self.vid[:, :, cell_id]
        trace = self.results['C'][cell_id, :]
        trace_raw = self.results['C_raw'][cell_id, :]

        cell_zoom = get_cell_zoom(cell)
        cell_zoom_contour = get_contour_fast(cell_zoom)
        cell_contour = get_contour_fast(cell)

        new_data = dict()
        new_data['xs'] = [cell_contour[:, 0]]
        new_data['ys'] = [cell_contour[:, 1]]
        new_data['fill_color'] = ["firebrick"]
        new_data['line_color'] = ["firebrick"]
        new_data['fill_alpha'] = self.rs3_contour.data['fill_alpha']
        new_data['line_alpha'] = self.rs3_contour.data['line_alpha']
        self.rs1_contour.data = new_data  # Updates data for plot

        new_data = dict()
        new_data['x'] = self.rs2_raw.data['x']
        new_data['y'] = trace_raw
        self.rs2_raw.data = new_data

        new_data = dict()
        new_data['x'] = self.rs2_deconv.data['x']
        new_data['y'] = trace
        self.rs2_deconv.data = new_data

        self.p2.title.text = str(cell_id) + '/' + str(self.ncells)
        self.p2.y_range.start = np.min(trace_raw)
        self.p2.y_range.end = np.max(trace_raw)

        new_data = dict()
        new_data['x'] = self.rs3_img.data['x']
        new_data['y'] = self.rs3_img.data['y']
        new_data['dh'] = self.rs3_img.data['dh']
        new_data['dw'] = self.rs3_img.data['dw']
        new_data['image'] = [cell_zoom]  # Must be a list
        self.rs3_img.data = new_data  # Updates data for plot
        self.p3.y_range.start = 0
        self.p3.y_range.end = cell_zoom.shape[0]
        self.p3.x_range.start = 0
        self.p3.x_range.end = cell_zoom.shape[1]

        new_data = dict()
        new_data['xs'] = [cell_zoom_contour[:, 0]]
        new_data['ys'] = [cell_zoom_contour[:, 1]]
        new_data['fill_color'] = ["firebrick"]
        new_data['line_color'] = ["firebrick"]
        new_data['fill_alpha'] = self.rs3_contour.data['fill_alpha']
        new_data['line_alpha'] = self.rs3_contour.data['line_alpha']
        self.rs3_contour.data = new_data  # Updates data for plot

        push_notebook(handle=self.handle)


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
    #     print(start_col, start_row, end_col, end_row,  max_col-(npixels/2),  max_row-(npixels/2))
    #     print(c0, r0)
    zoom_img[r0:r0 + end_row - start_row,
    c0:c0 + end_col - start_col] = img[start_row:end_row,
                                       start_col:end_col]

    return zoom_img


def normalize(trace):
    trace = trace - np.min(trace)
    if np.percentile(trace, 99) > 0:
        trace = trace / np.percentile(trace, 99)
    return trace


def get_contours(A, dims, thr=0.9):
    """Gets contour of spatial components and returns their coordinates
       From caiman.visualization

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     dims: tuple of ints
               Spatial dimensions of movie (x, y[, z])
    thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)

     Returns:
     --------
     Coor: list of coordinates with center of mass and
            contour plot coordinates (per layer) for each component
    """
    A = sparse.csc.csc_matrix(A)
    d, nr = np.shape(A)
    # if we are on a 3D video
    if len(dims) == 3:
        d1, d2, d3 = dims
        x, y = np.mgrid[0:d2:1, 0:d3:1]
    else:
        d1, d2 = dims
        x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = np.asarray([center_of_mass(a.toarray().reshape(dims, order='F'))
                    for a in A.T])

    # for each patches
    for i in range(nr):
        pars = dict()
        # we compute the cumulative sum of the energy of the
        # Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        cumEn = np.cumsum(patch_data[indx]**2)

        # we work with normalized values
        cumEn /= cumEn[-1]
        Bvec = np.ones(d)

        # we put it in a similar matrix
        Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        # for each dimensions we draw the contour
        for B in (Bmat if len(dims) == 3 else [Bmat]):
            # plotting the contour using an undocumented matplotlib
            # function around the threshold
            nlist = mpl._cntr.Cntr(y, x, B).trace(thr)

            # vertices will be the first half of the list
            vertices = nlist[:len(nlist) // 2]
            # this fix is necessary for having disjoint figures
            # and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for k, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = (np.round(old_div(vtx[-1, :], [d2, d1])) *
                                 [d2, d1])
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]),
                                             axis=0)

                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])),
                                   axis=0)

            pars['coordinates'] = (v if len(dims) == 2
                                   else (pars['coordinates'] + [v]))
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates
