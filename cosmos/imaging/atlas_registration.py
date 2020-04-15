#!/usr/bin/python
""""
A module for registering image coordinates to atlas coordinates.
"""

import os
import numpy as np
import scipy.io as spio
import pkg_resources
from tempfile import TemporaryFile

import cosmos.imaging.cell_selection as utils

import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import warnings


def load_atlas():
    """
    Load a top-view atlas, generated using the matlab functions
    GenerateAtlasImage.m (see this file for further details)
    and then copied into atlas folder.

    :return: atlas, annotations
    """
    path_name = pkg_resources.resource_filename(
        'cosmos', 'atlas/atlas_top_projection.mat')
    # print(path_name)
    mat = spio.loadmat(path_name, squeeze_me=True)
    atlas = mat['top_projection']
    atlas_outline = mat['clean_cortex_outline']

    # Collate annotations into a dict.
    acronyms = mat['acronyms']
    parents = mat['parents']
    names = mat['names']
    ids = mat['ids']
    annotations = dict()
    for (iter, id) in enumerate(ids):
        parent = parents[iter]
        if 'null' in parent:
            parent = None
        else:
            parent = int(parent)
        annotations[id] = {'acronym': acronyms[iter].encode('utf-8'),
                           'name': names[iter].encode('utf-8'),
                           'parent': parent}

    return atlas, annotations, atlas_outline


def get_parent_atlas(atlas, annotations, use_grandparent=True):
    parent_atlas = np.zeros(np.shape(atlas))
    id_list = np.unique(atlas)
    for (iter, id) in enumerate(id_list):
        if id > 0:
            parent = annotations[str(id)]['parent']
            if parent is not None:
                grandparent = annotations[str(parent)]['parent']
                if use_grandparent:
                    parent_atlas[atlas == id] = grandparent
                else:
                    parent_atlas[atlas == id] = parent

    return parent_atlas


def fit_atlas_transform(img_coords, atlas_coords):

    tform = utils.fit_transform(img_coords,
                                atlas_coords,
                                transform='similarity')
    return tform


def align_atlas_to_image(atlas, img, atlas_coords, img_coords, do_debug=False):
    """
    Warps an atlas image/outline to match a given image, based
    on provided keypoints.

    :param atlas: An image. Can be the outline, or actual atlas image.
    :param img: Image that you are aligning atlas to
               (this is used to crop atlas to correct size).
    :param atlas_coords: Keypoint coordinates on the atlas.
    :param img_coords: Keypoint coordinates on the image.
    :return: Atlas that has been warped to the image.
    """
    if do_debug:
        print('Obtaining alignment transform.')
    tform = fit_atlas_transform(img_coords, atlas_coords)
    aligned_coords = tform(atlas_coords)

    if do_debug:
        print('Image keypoints:')
        print(img_coords)
        print('Atlas keypoints:')
        print(aligned_coords)

    if do_debug:
        print('Aligning atlas.')
    aligned_atlas = utils.align_image(tform, atlas, output_shape=img.shape)
    aligned_atlas[np.where(aligned_atlas > 0)[0],
                  np.where(aligned_atlas > 0)[1]] = 1

    aligned_frames = utils.crop_images(img,
                                       aligned_atlas)
    aligned_img = aligned_frames[0]
    aligned_atlas = aligned_frames[1]

    if do_debug:
        plt.figure(),
        plt.subplot(121), plt.imshow(aligned_img),
        plt.plot(img_coords[:, 0], img_coords[:, 1], 'ro')

        plt.subplot(122), plt.imshow(aligned_atlas)
        plt.plot(aligned_coords[:, 0], aligned_coords[:, 1], 'ro')

    from skimage.morphology import skeletonize, binary_dilation, disk
    import scipy.signal
    filtered_atlas = scipy.signal.medfilt(aligned_atlas, 3)
    skeleton = skeletonize(filtered_atlas)
    outline = binary_dilation(skeleton, selem=disk(1))
    aligned_atlas = outline
    if do_debug:
        plt.figure()
        plt.imshow(outline)

    return aligned_atlas, aligned_img, tform

def get_atlas_outline(atlas):
    atlas[np.where(atlas > 0)[0],
                  np.where(atlas > 0)[1]] = 1
    from skimage.morphology import skeletonize, binary_dilation, disk
    import scipy.signal
    filtered_atlas = scipy.signal.medfilt(atlas, 3)
    skeleton = skeletonize(filtered_atlas)
    outline = binary_dilation(skeleton, selem=disk(1))
    return outline

def overlay_atlas_outline(atlas_outline, img):
    """
    Provided an aligned atlas outline and a target image
    display an overlay of the atlas on the image.

    :param atlas_outline: Image of atlas outline (aligned, same shape as img)
    :param img: Target image on which to overlay the
    :return: An image where the atlas outline pixels are set to the max value.
    """

    overlay = img.copy()
    overlay[np.where(atlas_outline > 0)[0],
            np.where(atlas_outline > 0)[1]] = np.max(overlay)

    return overlay


def region_name_from_coordinate(xy_coord, tform, atlas, annotations,
                                get_parent=False, do_debug=False):
    """
    Return the region name corresponding to a given roi coordinate.

    :param c: xy coordinate as a tuple
    :param tform: the skimage tform returned from align_atlas_to_image()
    :param atlas: the atlas loaded from load_atlas()
    :param annotations: the region annotations from load_atlas()
    :param get_parent: return the parent region name
    :param do_debug: make plots
    :return: Annotation dict entry corresponding to region id.
    """

    # Convert coordinate to atlas coordinates
    atlas_coords = tform.inverse(xy_coord)[0]
    atlas_coords = atlas_coords.astype(int)
    id = atlas[atlas_coords[1], atlas_coords[0]]
    if id == 0:
        annotation = None
        warnings.warn('Warning: There are likely neuronal sources located outside of the brain, maybe want to go back to  and cull them (i.e. in trace_merge_script.ipynb).')
    else:
        annotation = annotations[str(id)]

    if get_parent:
        if annotation is not None:
            child_annotation = annotation
            for i in range(2):  # Actually the grandparent is the proper level.
                parent_id = annotation['parent']
                if parent_id == 0:
                    annotation = None
                else:
                    annotation = annotations[str(parent_id)]
                    id = parent_id

    # Plot
    if do_debug:
        print('Child: ', child_annotation['name'])
        print(annotation['name'])
        if annotation is not None:
            plt.figure()
            plt.imshow(np.log(atlas + 1))
            plt.plot(atlas_coords[0], atlas_coords[1], 'ro')
            plt.title(annotation['name'])

    return annotation, id


def assign_cells_to_regions(xy_coords, tform, atlas, annotations,
                            get_parent=False, do_debug=False):
    """
    Given an array of xy_coordinates (i.e. the center of mass
    of each cell's ROI), return an array with the atlas region
    corresponding to each cell, as well as a dict which contains
    an entry for each region indicating the cells that are in that
    region.

    :param xy_coords: 2d ndarray. [X, Y]. Note: the center of mass
                     coordinates 'cm', as loaded may need to be flipped
                     i.e. using np.fliplr(cm)
    :param tform: alignment transform generated using
                  fit_atlas_transform(img_coords, atlas_coords)
    :param atlas: the unaligned atlas.
    :param annotations: dict. contains information about each id
                        in the atlas. Loaded using load_atlas(atlas_path).
    :param use_parent: bool. Return the higher level region category
                       as opposed to the specific subregion.
    :param do_debug: bool. Make debugging plots.
    :return: cells_in_region: dict. each key is a region id.
                              contains array of cell ids in that region.
    :return: region_of_cell: ndarray. contains the region id of each cell.
    """

    cells_in_region = dict()
    region_of_cell = []

    ncells = xy_coords.shape[0]
    for cell in range(ncells):
        xy_coord = xy_coords[cell, :]

        region, region_id = region_name_from_coordinate(xy_coord, tform,
                                                        atlas, annotations,
                                                        get_parent=get_parent,
                                                        do_debug=False)
        region_of_cell.append(region_id)

        # Initialize dict entry, or append to existing entry.
        if region_id in cells_in_region.keys():
            cells_in_region[region_id].append(cell)
        else:
            cells_in_region[region_id] = [cell]

    return cells_in_region, region_of_cell


from tkinter import Tk, Label, Button, Canvas, PhotoImage, LEFT, RIGHT, W
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class AlignAtlasGUI:
    """
    A GUI that allows finescale translation and scaling
    to align an atlas to an image.

    Uses as input the output from load_atlas() and select_keypoints().
    """
    def __init__(self, master,  atlas_outline, img,
                 atlas_coords, img_coords, save_loc):
        self.master = master
        master.title("Atlas alignment GUI")
        master.protocol("WM_DELETE_WINDOW", self.disable_event) #Disable 'close' button.
                                                                # Use EXIT button instead.
        self.atlas_outline = atlas_outline
        self.img = img
        self.atlas_coords = atlas_coords
        self.img_coords = img_coords
        self.save_loc = save_loc # Temporary location to save out aligned img_coords

        # Sets the magnitude of each translation or scaling button press.
        self.shift_size = 1
        self.scale_size = 1

        # Set up the GUI layout.
        self.label = Label(master, text="Translate!")
        self.label.grid(row=1, column=0)

        self.translateL_button = Button(master, text="<", command=self.translateL)
        self.translateL_button.grid(row=1, column=1)
        self.translateR_button = Button(master, text=">", command=self.translateR)
        self.translateR_button.grid(row=1, column=2)
        self.translateU_button = Button(master, text="^", command=self.translateU)
        self.translateU_button.grid(row=1, column=3)
        self.translateD_button = Button(master, text="D", command=self.translateD)
        self.translateD_button.grid(row=1, column=4)
        self.label1 = Label(master, text="Scale:")
        self.label1.grid(row=1, column=5)
        self.scaleU_button = Button(master, text="^", command=self.scaleU)
        self.scaleU_button.grid(row=1, column=6)
        self.scaleD_button = Button(master, text="D", command=self.scaleD)
        self.scaleD_button.grid(row=1, column=7)
        self.exit_button = Button(master, text="Finished (EXIT)", command=self.save_and_exit)
        self.exit_button.grid(row=1, column=8, columnspan=50)

        # Generate initial alignment.
        aligned_atlas_outline, aligned_img, tform = align_atlas_to_image(
            self.atlas_outline, self.img, self.atlas_coords, self.img_coords)
        overlay = overlay_atlas_outline(aligned_atlas_outline, self.img)

        # Generate figure handles to be updated after each button press.
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        self.imdisplay = ax.imshow(overlay)
        ax.plot(self.img_coords[:, 0], self.img_coords[:, 1], 'ro')
        self.keypointdisplay, = ax.plot(self.img_coords[:,0],
                                        self.img_coords[:, 1], 'bo')
        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=2, columnspan=50, rowspan=50)
        self.update_canvas()

    def disable_event(self):
        """
        Used for disabling the 'close' button.
        """
        pass

    def update_canvas(self):
        """
        Updates the alignment based on the current state of img_coords.
        Overlays the aligned atlas on the image and updates the display.
        """
        aligned_atlas_outline, aligned_img, tform = align_atlas_to_image(
            self.atlas_outline, self.img, self.atlas_coords, self.img_coords)
        overlay = overlay_atlas_outline(aligned_atlas_outline, self.img)
        self.imdisplay.set_data(overlay)
        self.keypointdisplay.set_xdata(self.img_coords[:,0])
        self.keypointdisplay.set_ydata(self.img_coords[:,1])
        self.canvas.draw()

    def translateR(self):
        self.img_coords[:, 0] += self.shift_size
        print("Translate right.")
        self.update_canvas()

    def translateL(self):
        self.img_coords[:, 0] -= self.shift_size
        print("Translate left.")
        self.update_canvas()

    def translateD(self):
        self.img_coords[:, 1] += self.shift_size
        print("Translate down.")
        self.update_canvas()

    def translateU(self):
        self.img_coords[:, 1] -= self.shift_size
        print("Translate up.")
        self.update_canvas()

    def scaleU(self):
        diff = self.img_coords[0,:] - self.img_coords[1,:]
        diff = diff/np.sqrt(np.sum(diff**2))
        scale = self.scale_size
        self.img_coords[0,:] += scale*diff
        self.img_coords[1,:] -= scale*diff
        print("Scale up.")
        self.update_canvas()

    def scaleD(self):
        diff = self.img_coords[0,:] - self.img_coords[1,:]
        diff = diff/np.sqrt(np.sum(diff**2))
        scale = self.scale_size
        self.img_coords[0,:] -= scale*diff
        self.img_coords[1,:] += scale*diff
        print("Scale down.")
        self.update_canvas()

    def save_and_exit(self):
        print("Saving.")
        np.savez(self.save_loc, img_coords=self.img_coords,
                 atlas_coords=self.atlas_coords)

        print("Exiting.")
        self.master.quit()  # stops mainloop
        self.master.destroy()

def select_keypoints(atlas_outline, img):
    """
    Presents an interface for selecting two
    keypoints (along the midline) that define
    the initial position (and importantly, the
    rotation) of the aligned atlas.

    :param atlas_outline: Outline to overlay on atlas.
    :param img: Image to which the atlas will be aligned.
    :return: atlas_coords_array: the keypoints in atlas space
             img_coords_array: the corresponding keypoints in image space
                               (which were user selected).
    """
    keypoint_positions = ['Anterior midline',
                          'Posterior midline']

    ### Load atlas.
    # atlas, annotations, atlas_outline = load_atlas()

    do_manual_atlas_keypoint = False
    if do_manual_atlas_keypoint:
        atlas_coords = []
        plt.figure(figsize=(30, 30))
        plt.imshow(atlas_outline)
        for t in keypoint_positions:
            plt.title('Click on ' + t)
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            atlas_coords.extend(c)
    else:
        atlas_coords = [(98, 227),  # (83, 227)
                        (348, 227)]
                       ### These numbers were taken from an initial manual selection.
        plt.figure(figsize=(30, 30))
        plt.imshow(atlas_outline)
        for cc in atlas_coords:
            plt.plot(cc[0], cc[1], 'ro')

    ### Convert selected keypoints to array.
    atlas_coords_array = np.zeros((len(atlas_coords), 2))
    for ci, c in enumerate(atlas_coords):
        atlas_coords_array[ci, 0] = np.round(c[0])
        atlas_coords_array[ci, 1] = np.round(c[1])

    while 1:
        plt.figure()
        plt.imshow(img)
        img_coords = []
        do_manual_patch_keypoint = True
        if do_manual_patch_keypoint:
            plt.figure(figsize=(30, 30))
            plt.imshow(img)
            for t in keypoint_positions:
                plt.title('Click on ' + t)
                c = plt.ginput(n=1)
                plt.plot(c[0][0], c[0][1], 'ro')
                img_coords.extend(c)
        else:
            ### These numbers are just for fast debugging.
            img_coords = [(26, 297),
                          (430, 314)]
            # (26, 187),
            # (17, 420)]
            plt.figure(figsize=(30, 30))
            plt.imshow(img, cmap='Greys_r')
            for cc in img_coords:
                plt.plot(cc[0], cc[1], 'ro')

        plt.close('all')

        ### Convert selected keypoints to array.
        img_coords_array = np.zeros((len(img_coords), 2))
        for ci, c in enumerate(img_coords):
            img_coords_array[ci, 0] = np.round(c[0])
            img_coords_array[ci, 1] = np.round(c[1])

        return atlas_coords_array, img_coords_array


def run_align_atlas_gui(img=None, fig_path=None):
    """
    Initializes a GUI interface for aligning
    an atlas to an underlying image.

    :param img: np.array. Image to align the atlas to.
    :param fig_path: Optionally provide this to save out
                     plots of the aligned atlas.
    :return: atlas_coords: keypoints on the atlas (these stay constant)
             img_coords: keypoints on the image that define the alignment.
             aligned_atlas_outline: For overlaying on image.
             atlas: The actual (unaligned) atlas (from load_atlas())
    """
    if img is None:
        # For debugging.
        img = np.zeros((500, 500))
        img[10, 10] = 1

    atlas, annotations, atlas_outline = load_atlas()

    ### Initialize keypoints selection.
    atlas_coords_array, img_coords_array = select_keypoints(atlas_outline, img)
    atlas_coords = atlas_coords_array[0:2,:]
    img_coords = img_coords_array[0:2,:]

    aligned_atlas_outline, aligned_img, tform = align_atlas_to_image(
        atlas_outline, img,
        atlas_coords,
        img_coords,
        do_debug=False
    )
    overlay = overlay_atlas_outline(aligned_atlas_outline, img)

    ### Launch GUI for refining the initial alignment.
    save_loc = TemporaryFile()
    root = Tk()
    my_gui = AlignAtlasGUI(root, atlas_outline, img, atlas_coords, img_coords, save_loc)
    root.mainloop()

    print("Aligned.")
    save_loc.seek(0)
    aligned_out = np.load(save_loc)
    atlas_coords = aligned_out['atlas_coords']
    img_coords = aligned_out['img_coords']

    # Save some summary plots.
    aligned_atlas_outline, aligned_img, tform = align_atlas_to_image(
        atlas_outline, img,
        atlas_coords,
        img_coords,
        do_debug=False
    )
    overlay = overlay_atlas_outline(aligned_atlas_outline, img)

    plt.figure()
    plt.imshow(img, cmap='Greys_r')
    for ci, c in enumerate(img_coords):
        plt.plot(c[0], c[1], 'ro')
    if fig_path is not None:
        plt.savefig(fig_path + '_keypoints.png')

    plt.figure()
    plt.imshow(overlay, cmap='Greys_r')
    if fig_path is not None:
        plt.savefig(fig_path + '_overlay.png')

    return atlas_coords, img_coords, aligned_atlas_outline, atlas


def run_align_atlas_gui_simple(img, atlas, fig_path):
    """
    This is DEPRECATED code as of 20180709, replaced by run_align_atlas_gui.
    Potentially to be deleted eventually.

    To add:
    Simple scaling and translation, based on the original two keypoints (which set the orientation).
    With real time update/feedback.

    :param img:
    :param atlas:
    :param fig_path:
    :return:
    """
    warning('run_align_atlas_gui_simple is DEPRECATED.')

    keypoint_positions = ['Anterior midline',
                          'Posterior midline']
    # 'Right anterolateral corner',
    # 'Left anterolateral corner']

    ### Load atlas.
    atlas, annotations, atlas_outline = load_atlas()
    do_manual_atlas_keypoint = False
    if do_manual_atlas_keypoint:
        atlas_coords = []
        plt.figure(figsize=(30, 30))
        plt.imshow(atlas_outline)
        for t in keypoint_positions:
            plt.title('Click on ' + t)
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            atlas_coords.extend(c)
    else:
        atlas_coords = [(98, 227),  # (83, 227)
                        (348, 227)]
        # (83, 303),
        # (83, 151)] ### These numbers were taken from an initial manual selection.
        plt.figure(figsize=(30, 30))
        plt.imshow(atlas_outline)
        for cc in atlas_coords:
            plt.plot(cc[0], cc[1], 'ro')

    ### Convert selected keypoints to array.
    atlas_coords_array = np.zeros((len(atlas_coords), 2))
    for ci, c in enumerate(atlas_coords):
        atlas_coords_array[ci, 0] = np.round(c[0])
        atlas_coords_array[ci, 1] = np.round(c[1])

    while 1:
        plt.figure()
        plt.imshow(img)
        img_coords = []
        do_manual_patch_keypoint = True
        if do_manual_patch_keypoint:
            plt.figure(figsize=(30, 30))
            plt.imshow(img)
            for t in keypoint_positions:
                plt.title('Click on ' + t)
                c = plt.ginput(n=1)
                plt.plot(c[0][0], c[0][1], 'ro')
                img_coords.extend(c)
        else:
            ### These numbers are just for fast debugging.
            img_coords = [(26, 297),
                          (430, 314)]
            # (26, 187),
            # (17, 420)]
            plt.figure(figsize=(30, 30))
            plt.imshow(img, cmap='Greys_r')
            for cc in img_coords:
                plt.plot(cc[0], cc[1], 'ro')

        plt.close('all')

        ### Convert selected keypoints to array.
        img_coords_array = np.zeros((len(img_coords), 2))
        for ci, c in enumerate(img_coords):
            img_coords_array[ci, 0] = np.round(c[0])
            img_coords_array[ci, 1] = np.round(c[1])

        aligned_atlas_outline, aligned_img, tform = align_atlas_to_image(
            atlas_outline, img,
            atlas_coords_array[0:2, :],
            img_coords_array[0:2, :],
            do_debug=False
        )

        ### Overlay atlas on image for checking that things look good.
        overlay = overlay_atlas_outline(aligned_atlas_outline, img)
        plt.figure(figsize=(20, 20))
        plt.imshow(overlay, cmap='Greys_r')
        plt.title(
            'Check that things look good, and close this window manually.')
        plt.show()

        text = input('Look good? [y] or [n]')
        print(text)
        if text == 'y':
            break

    # ### Save out selections
    # keypoints_dir = os.path.join(self._keypoints_file, name, \
    #                              str(name) + '_source_extraction')
    # save_fname = os.path.join(keypoints_dir, 'keypoints.npz')
    # print('Saving keypoints and aligned atlas to: ' + save_fname)
    # np.savez(save_fname,
    #          coords=img_coords_array,
    #          atlas_coords=atlas_coords,
    #          atlas=atlas,
    #          img=aligned_img,
    #          aligned_atlas_outline=aligned_atlas_outline)

    plt.figure()
    plt.imshow(img, cmap='Greys_r')
    for ci, c in enumerate(img_coords):
        print(c)
        plt.plot(c[0], c[1], 'ro')
    plt.savefig(fig_path + '_keypoints.png')

    plt.figure()
    plt.imshow(overlay, cmap='Greys_r')
    plt.savefig(fig_path + '_overlay.png')

    ### Do these need to be 'array's?
    return (atlas_coords_array, img_coords_array, aligned_atlas_outline, atlas)



# TODO: Align a video to the atlas and overlay atlas (for display purposes
# - don't do general processing with this). Potentially just faster to do this
# in Keynote...
