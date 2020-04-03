"""
Test that image loading in python2.7 works as expected.
This is not a formal unit test because it is not expected
to work in the same environment as the other tests,
which are for python3.
"""
import os
from cosmos.imaging.frame_processor import FrameProcessor
import cosmos.imaging.img_io as iio


def test_load():
    d = {'date': '20190604', 'name': 'example_data_6frames'}

    test_dir, _ = os.path.split(__file__)
    test_dir = os.path.join(test_dir, 'test_raw_data')
    processed_data_dir = test_dir

    f = FrameProcessor(raw_data_dir=test_dir,
                       processed_data_dir=processed_data_dir,
                       dataset=d)

    stack = iio.load_raw_data(f._data_path, sub_seq=range(3), print_freq=1)
    assert(stack.shape[0] == 3)
    assert(stack.shape[1] == 791)
    assert(stack.shape[2] == 1607)
    assert(stack[0, 0, 0] == 5309)

    stack = iio.load_raw_data(f._data_path, print_freq=1)
    assert(stack.shape[0] == 6)
    assert(stack.shape[1] == 791)
    assert(stack.shape[2] == 1607)
    assert(stack[0, 0, 0] == 5309)

if __name__ == '__main__':
    test_load()