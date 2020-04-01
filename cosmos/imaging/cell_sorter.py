from IPython.display import display
from ipywidgets import widgets
from skimage import measure
import numpy as np

from bokeh.io import output_file, show, output_notebook, push_notebook
from bokeh.layouts import widgetbox, column, row, layout
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import Range1d


class CellSorter():
    """
    A jupyter notebook GUI using bokeh for stepping through a
    collection of sources extracted by CNMF,
    displaying their ROI, raw, and deconvolved traces.
    A text box is provided to indicate whether a cell should be
    kept (type 'k'), or deleted (type 'd').
    Type 'b' to go back one cell.
    To save the array at any time, type in 's'.
    You MUST type 's' in order to save your results for further use.

    :param footprints: a 3D np.array [X x Y x nROIs] of cell footprints
    :param traces: 2D np.array [cell x T] with smoothed traces for each cell
    :param traces_raw: 2D np.array [cell x T] with raw traces for each cell
    :param base_im: 2D np.array background image used when overlaying ROIs
    :param save_path: Path for saving out the array of which cells to keep
    :param keep_cells: An array, passed in by reference, which contains the
                       labels (-1 = not yet labeled, 1 = keep, 0 = delete)
                       for each cell (this can include the auto-processed labels).
                       This array will be modified in place.
    :param show_only_kept_cells: (optional). bool.
                                 only display cells where the keep_cells array is 1.
                                 (this enables you to go through and double check those).
    """
    def __init__(self, footprints, traces,
                 traces_raw, base_im,
                 save_path, keep_cells,
                 show_only_kept_cells=True):

        self.footprints = footprints
        self.traces = traces
        self.traces_raw = traces_raw
        self.base_im = base_im
        self.save_path = save_path
        self.keep_cells = keep_cells
        self.show_only_kept_cells = show_only_kept_cells

        output_notebook()
        self._initialize_plot()

    def _initialize_plot(self):
        """
        Sets up GUI interface that displays
        a cell contour overlaid on full field of view,
        a zoom in of the cell contour,
        raw and smoothed traces,
        a text box input widget,
        a slider widget.
        """
        print('Initializing CellSorter class.')
        self.ncells = self.footprints.shape[2]
        if self.show_only_kept_cells:
            self.which_cells = np.where(self.keep_cells > 0)[0]
        else:
            self.which_cells = np.arange(len(self.keep_cells))

        cell_iter = 0
        cell_id = self.which_cells[cell_iter]
        cell = self.footprints[:, :, cell_id]
        trace = self.traces[cell_id, :]
        trace_raw = self.traces_raw[cell_id, :]

        cell_zoom = self._get_cell_zoom(cell)
        cell_zoom_contour = self._get_contour_fast(cell_zoom)
        cell_contour = self._get_contour_fast(cell)

        if cell_contour is None:
            cell_contour = np.array([[0, 1], [0, 1]])

        if cell_zoom_contour is None:
            cell_zoom_contour = np.array([[0, 1], [0, 1]])

        # Initialize plots.
        # Cell contour on full image.
        self.p1 = figure(plot_width=300, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        img = self.base_im
        self.r1_img = self.p1.image(image=[img],
                                    x=[0], y=[0],
                                    dw=[img.shape[1]],
                                    dh=[img.shape[0]])
        r1_contour = self.p1.patches(xs=[cell_contour[:, 0]],
                                     ys=[cell_contour[:, 1]],
                                     color=["firebrick"],
                                     fill_alpha=[0.0],
                                     line_alpha=[1.0],
                                     line_width=2)
        self.p1.x_range = Range1d(0, img.shape[0])
        self.p1.y_range = Range1d(0, img.shape[1])

        # Zoom in of cell contour.
        self.p3 = figure(plot_width=300, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        img = cell_zoom
        r3_img = self.p3.image(image=[img],
                               x=[0], y=[0],
                               dw=[img.shape[1]],
                               dh=[img.shape[0]])
        r3_contour = self.p3.patches(xs=[cell_zoom_contour[:, 0]],
                                     ys=[cell_zoom_contour[:, 1]],
                                     color=["firebrick"],
                                     fill_alpha=[0.0],
                                     line_alpha=[1.0],
                                     line_width=2)
        self.p3.x_range = Range1d(0, img.shape[0])
        self.p3.y_range = Range1d(0, img.shape[1])

        # Cell footprint. This is currently not displayed.
        self.p4 = figure(plot_width=300, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        r4 = self.p4.image(image=[cell],
                           x=[0], y=[0],
                           dw=[1], dh=[1])
        self.p4.x_range = Range1d(0, 1)
        self.p4.y_range = Range1d(0, 1)

        # Raw and smoothed trace overlaid.
        self.p2 = figure(plot_width=900, plot_height=300,
                         toolbar_location="above",
                         tools="pan,wheel_zoom,box_zoom,reset")
        nx = len(trace)
        r2_raw = self.p2.line(x=range(nx),
                              y=trace_raw,
                              line_width=2,
                              color='firebrick')
        r2_deconv = self.p2.line(x=range(nx),
                                 y=trace,
                                 line_width=2,
                                 color='blue')
        self.p2.x_range = Range1d(0, nx)
        self.p2.y_range = Range1d(np.min(trace_raw),
                                  np.max(trace_raw))

        # Handles for updating data.
        self.rs1_contour = r1_contour.data_source
        self.rs2_raw = r2_raw.data_source
        self.rs2_deconv = r2_deconv.data_source
        self.rs3_img = r3_img.data_source
        self.rs3_contour = r3_contour.data_source
        self.rs4 = r4.data_source

        # Add widgets (slider and text box).
        self.w1 = widgets.Text(value='',
                               placeholder='[k]eep, [d]elete, [s]ave, [b]ack',
                               description='[k]eep/[d]elete/[s]ave/[b]ack? ',
                               disabled=False,
                               continuous_update=True)

        self.w2 = widgets.IntSlider(value=0,
                                    min=0,
                                    # max=self.ncells-1,
                                    max=len(self.which_cells)-1,
                                    step=1,
                                    description='Frame',
                                    disabled=False,
                                    continuous_update=True,  # True for smoother transition
                                                             # (but potentially laggy).
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='d')

        # Render plots and widgets.
        self.handle = show(column(row(self.p1, self.p3),
                                  self.p2),
                           notebook_handle=True)
        display(self.w1, self.w2)

        self._update_cell(cell_id)

        # Link up widget callbacks.
        self.w1.on_submit(self._handle_submit)
        # self.w1.observe(self._handle_submit, 'value')  # w1.observe does not enable
                                                         # accepting a source by
                                                         # just hitting enter in the gui.
                                                         # So, until on_submit is
                                                         # fully deprecated,
                                                         # just use on_submit.
        self.w2.observe(self._handle_slider, names='value')

    def _handle_submit(self, sender):
        cell_iter = self.w2.value
        cell_id = self.which_cells[cell_iter]
        if self.w1.value == 'k':
            self.keep_cells[cell_id] = 1
        elif self.w1.value == 'd':
            self.keep_cells[cell_id] = 0
        elif self.w1.value == 'b':
            if cell_iter > 1:
                cell_iter = cell_iter - 2
        elif self.w1.value == 's':
            if self.save_path is not None:
                np.savez(self.save_path,
                         keep_cells=self.keep_cells)
                np.savez(self.save_path + '.manual_backup.npz',
                         keep_cells=self.keep_cells)
                print('Saved to: ' + self.save_path)
                self.w1.value = ''  # Reset the text box to empty
                self.w2.value = cell_iter
                return

        cell_iter = cell_iter + 1
        if cell_iter < len(self.which_cells):
            cell_id = self.which_cells[cell_iter-1]
            self._update_cell(cell_id)
        else:
            print('All cells completed.')

        self.w1.value = ''  # Reset the text box to empty
        self.w2.value = cell_iter

    def _handle_slider(self, change):
        cell_iter = change.new
        cell_id = self.which_cells[cell_iter]
        self._update_cell(cell_id)

    def _update_cell(self, cell_id):
        # Note: when updating data in bokeh, need to be sure
        # to update ALL of the fields, or it may not work.

        colors = ["orange", "firebrick", "green"]
        cell_color = colors[int(self.keep_cells[int(cell_id)] + 1)]

        cell = self.footprints[:, :, cell_id]
        trace = self.traces[cell_id, :]
        trace_raw = self.traces_raw[cell_id, :]

        cell_zoom = self._get_cell_zoom(cell)
        cell_zoom_contour = self._get_contour_fast(cell_zoom)
        cell_contour = self._get_contour_fast(cell)

        new_data = dict()
        if cell_contour is not None:
            new_data['xs'] = [cell_contour[:, 0]]
            new_data['ys'] = [cell_contour[:, 1]]
        else:
            new_data['xs'] = [1]
            new_data['ys'] = [2]
        new_data['fill_color'] = [cell_color]
        new_data['line_color'] = [cell_color]
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

        self.p2.title.text = str(cell_id) + '/' + str(self.ncells-1)
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
        if cell_zoom_contour is not None:
            new_data['xs'] = [cell_zoom_contour[:, 0]]
            new_data['ys'] = [cell_zoom_contour[:, 1]]
        else:
            new_data['xs'] = [1]
            new_data['ys'] = [2]
        new_data['fill_color'] = [cell_color]
        new_data['line_color'] = [cell_color]
        new_data['fill_alpha'] = self.rs3_contour.data['fill_alpha']
        new_data['line_alpha'] = self.rs3_contour.data['line_alpha']
        self.rs3_contour.data = new_data  # Updates data for plot

        push_notebook(handle=self.handle)

    @staticmethod
    def _get_contour_fast(cell, level=0.1):
        """
        Returns the largest cell ROI contour in the
        provided image. Output can be plotted with bokeh using:
            p.patches(xs=[contour[:,0]], ys=[contour[:,1]],
                      color=["firebrick"], fill_alpha=[0.0],
                      line_alpha = [1.0], line_width=2)
        :param cell: an image containing the ROI of a cell.
        :param level: fraction of max value at which to find contour.
        :return contour: (2d ndarray)
        """
        contours = measure.find_contours(cell, level * np.amax(cell))
        if len(contours) > 0:
            contour = contours[np.argmax([len(x) for x in contours])]
            contour = np.fliplr(contour)
            return contour
        else:
            return None

    @staticmethod
    def _get_cell_zoom(img, npixels=100):
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
