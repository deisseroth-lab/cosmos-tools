import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import cosmos.reg as reg

#### This is a test script to check that you can extract a region from an atlas
#### when providing the coordinates of the center of mass of an ROI.

if __name__ == "__main__":
    # atlas_loc = '/home/izkula/Dropbox/cosmos_data/atlas/atlas_top_projection.mat'
    atlas, annotations, atlas_outline = reg.load_atlas()

    keypoints_file =  '/home/izkula/Data/processedData/20180227/cux2m72_COSMOSTrainMultiBlockGNG_1/top/top_source_extraction/keypoints.npz'
    keypoints = np.load(keypoints_file)

    img = keypoints['img']
    atlas = keypoints['atlas']
    img_coords = keypoints['coords']
    atlas_coords = keypoints['atlas_coords']
    aligned_atlas_outline = keypoints['aligned_atlas_outline']

    tform = reg.fit_atlas_transform(img_coords, atlas_coords)


    xy_coord = [320, 450]
    plt.figure()
    plt.imshow(aligned_atlas_outline)
    plt.plot(xy_coord[0], xy_coord[1], 'ro')
    reg.region_name_from_coordinate(xy_coord, tform, atlas, annotations, get_parent=True, do_debug=True)

    plt.show()