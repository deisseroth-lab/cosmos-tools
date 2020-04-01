import numpy as np
import os
from cosmos.imaging.frame_processor import FrameProcessor


def test_get_motion():
    """
    Ensure that relative motion calculation is correct.
    Also test that shift_frame() to perform the motion
    correction is accurate.
    """

    f = FrameProcessor(raw_data_dir='test',
                       processed_data_dir='test',
                       dataset={'date':'20180101', 'name':'test'})

    template = np.zeros((20, 30))
    template[10, 10] = 1
    frame = np.roll(template, 5, axis=0)
    frame = np.roll(frame, -2, axis=1)

    crop = np.stack((template, frame), axis=0)
    shiftx, shifty, template_crop, target_crop = f.get_motion(crop)
    shiftx = np.mean(shiftx[1])
    shifty = np.mean(shifty[1])

    assert(shifty == 5)
    assert(shiftx == -2)

    shifted = f.shift_frame(frame, shiftx, shifty)
    assert(np.all(shifted == template))
