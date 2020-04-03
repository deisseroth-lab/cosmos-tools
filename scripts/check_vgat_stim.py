import os
import tifffile
import matplotlib.pyplot as plt
import cosmos.imaging.img_io as iio
from skimage.external.tifffile import imread

if __name__ == "__main__":
    raw_data_dir = "/media/optodata2/tam/"

    input_folder = os.path.join(raw_data_dir, '20190112')
    tif_files = ['vGatm15-20190112-173522-upper-camera.tif']
    sub_seq = range(1000)
    # stack = skimage.external.tifffile.imread(os.path.join(input_folder, tif_files[0]), pages=range(1000))
    stack = imread(os.path.join(input_folder, tif_files[0]), pages=range(100))

    # print('Loading: ', input_folder, tif_files[0])
    # with tifffile.TiffFile(os.path.join(input_folder,
    #                                     tif_files[0])) as tif:
    #     stack = tif.asarray(key=sub_seq)
    #     # stack = tif.asarray()
    print(stack.shape)
