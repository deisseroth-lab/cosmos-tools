import os
import numpy as np
from cosmos.imaging.cosmos_dataset import CosmosDataset


def test_load():
    # Note: These test datasets are generated using the matlab script generate_test_dataset.m
    dataset = {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.splitext(filename)
    CD = CosmosDataset(test_dir, dataset, fig_save_path=None)
    print(CD.data['top_focus']['path'])

    assert(CD.data['top_focus']['results']['C'].shape == (10, 500))

    do_load = False
    do_manual = False
    CD.cull(do_auto=not do_load, do_load=do_load, do_manual=do_manual, which_key='top_focus')
    CD.cull(do_auto=not do_load, do_load=do_load, do_manual=do_manual, which_key='bot_focus')

    CD.align_planes(use_culled_cells=True)
    CD.merge_planes(do_debug=False)
    save_path = CD.save_merged()
    assert(os.path.exists(save_path))
    os.remove(save_path)


def test_nan_interp():
    dataset = {'date': '20180227', 'name': 'testdata_cux2m72_COSMOSTrainMultiBlockGNG_1'}
    filename = __file__
    test_dir, _ = os.path.splitext(filename)
    CD = CosmosDataset(test_dir, dataset, fig_save_path=None)

    y = np.array([1, np.nan, 1])
    assert(np.array_equal(CD._interp_nans(y), np.array([1, 1, 1])))

    y = np.array([1, np.nan, 0.5])
    assert(np.array_equal(CD._interp_nans(y), np.array([1, 0.75, 0.5])))

    y = np.array([np.nan, np.nan, np.nan])
    assert(np.isnan(CD._interp_nans(y)).all())

    y = np.array([1, 1, 0.5])
    assert(np.array_equal(CD._interp_nans(y), np.array([1, 1, 0.5])))

    y = np.array([1, 1, 1, np.nan, 0.5])
    assert(np.array_equal(CD._interp_nans(y), np.array([1, 1, 1, 0.75, 0.5])))



## TODO:
## Possible example tests (from the ipynb)
#### Testing code for CD.align_planes()
#### The cm's line up with the aligned images.
"""
cellid=100
key = 'bot_focus'
img = CD.footprints_aligned[key][:,:,cellid]
cm = CD.cm_aligned[key][cellid,:]

xr = models.Range1d(start=cm[1]-20, end=cm[1]+20)
yr = models.Range1d(start=cm[0]-20, end=cm[0]+20)
p1 = figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)
r1_img = p1.image(image=[img], x=[0], y=[0], dw=[img.shape[1]], dh=[img.shape[0]])
show(p1)
print(cm)

img = CD.footprints_unalign[key][:,:,cellid]
cm = CD.cm_unalign[key][cellid,:]

xr = models.Range1d(start=cm[1]-20, end=cm[1]+20)
yr = models.Range1d(start=cm[0]-20, end=cm[0]+20)
p2 = figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)
r2_img = p2.image(image=[img], x=[0], y=[0], dw=[img.shape[1]], dh=[img.shape[0]])
show(p2)
print(cm)

### Testing code for CD.align_planes().
### Here - after alignment, the keypoints
### and cm should have moved in the same direction.
### The magenta keypoints in each image should match. 

doPlotOrig = True
doPlotAligned = True
doPlotCM = True

if doPlotOrig:
    plt.figure(figsize=(15,15))
    for ind, key in enumerate(['top_focus', 'bot_focus']):
        plt.subplot(1,2,ind+1)
        plt.imshow(CD.mean_frames_unalign[key])

if doPlotAligned:
    plt.figure(figsize=(15,15))
    for ind, key in enumerate(['top_focus', 'bot_focus']):
        plt.subplot(1,2,ind+1)
        plt.imshow(CD.mean_frames_aligned[key], clim=[0, 40000])
        plt.plot(CD.keypoints_unalign[key][:,0], 
                 CD.keypoints_unalign[key][:,1], 'co')
        plt.plot(CD.keypoints_aligned[key][:,0], 
                 CD.keypoints_aligned[key][:,1], 'mo')
        if doPlotCM:
            plt.plot(CD.cm_unalign[key][:,1], CD.cm_unalign[key][:,0], 'bo')
            plt.plot(CD.cm_aligned[key][:,1], CD.cm_aligned[key][:,0], 'ro')
"""