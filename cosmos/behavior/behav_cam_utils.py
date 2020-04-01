import numpy as np
import matplotlib.pyplot as plt


### TODO: Fix load_ttl_times. Double check the resampling to match imaging speeds.

### OUTPUT TO [trials x time] for each covariate.


def get_led_frames(stack, yslice, xslice,
                   save_path=None,
                   do_plot=True,
                   do_display_LED_frames=False):
    """
    Extract the frame number of each trial start, as indicated
    by a green LED flooding the frame.

    :param stack: [nframes x NY x NX x ncolorchannels]
    :param yslice: vertical roi crop coordinates as a slice object, i.e. slice(10, 20)
    :param xslice: horizontal roi crop coordinates as a slice object, i.e. slice(10, 20)
    :param save_path: optionally save out led frames and crop region trace.
    :param do_plot: bool. Plot the led frames overlaid on the roi trace.
    :param do_display_LED_frames: bool. Show each frame to double check that the green light is on.
                                  Takes a while.
    :returns led_frames: a list of frame indices.
    """

    tslice = slice(0, None)
    cslice = slice(0, None)
    roi = (tslice, yslice, xslice, cslice)
    #     zeros_mask = np.zeros_like(stack[0], np.uint8)
    #     zeros_mask[roi[1:]] = 1
    #     plt.imshow(stack[20000]*zeros_mask)
    #     plt.xlabel('x');
    #     plt.ylabel('y');

    rgb_file = save_path + 'RGB.npy'
    #     if not op.exists(rgb_file):
    rgb = stack[roi].sum(1).sum(1)
    np.save(rgb_file, rgb)
    #     else:
    #         lower_rgb = np.load(lower_rgb_file)

    if do_plot:
        plt.figure(figsize=(20, 5))
        plt.plot(rgb[:, 0], 'r')
        plt.plot(rgb[:, 1], 'g')
        plt.plot(rgb[:, 2], 'b')

    ### Extract LED frames
    green_thresh = rgb[:, 1].mean() + 10 * np.std(rgb[:, 1])
    green_frames = np.where(rgb[:, 1] > green_thresh)[0]
    led_frames = green_frames[
        np.where(np.diff(green_frames) > 1)[0]]  # Exclude consecutive indices
    if do_plot:
        plt.figure(figsize=(40, 5))
        plt.plot(led_frames, rgb[led_frames, 1], 'ro')
        plt.plot(rgb[:, 1], 'g')
    print('Detected %i frames above threshold.' % (led_frames.shape[0]))
    np.save(save_path + 'led_frames.npy', led_frames)

    if do_display_LED_frames:
        for tt in led_frames:
            plt.figure()
            plt.imshow(stack[tt])
            plt.title(tt)

    # # TODO
    # do_remove_false_positives = False
    # if do_remove_false_positives:
    #     pass
    #     false_positive_led_frames = [8618]
    # lower_trial_on_led = np.delete(lower_trial_on_led, [0]) # #mouse = 'cux2m943' # date = '20180424'
    # np.save(op.join(save_dir,  '%s_%s_1_lower_trial_on_led.npy' % (d['date'], d['name'])), lower_trial_on_led)

    return led_frames


def get_motion_energy(roi, data, do_normalize=True):
    """
    Returns the overall changes in pixel value within
    the specified roi, between successive frames of data.

    :param roi: a tuple of slices, with n components
                where n is the number of dimensions of data.
                i.e. (slice(15, 45), slice(0, 25), None)
    :param data: [nframes x NY x NX x Ncolorchannels]

    :return energy: [nframes]. The motion energy for each
                    frame.
    """

    diff = data[:-2, roi[0], roi[1]] - data[1:-1, roi[0], roi[1]]
    energy = np.sqrt((diff**2).sum(1).sum(1).sum(1))
    if do_normalize:
        roi_npix = (roi[0].stop - roi[0].start)*(roi[1].stop - roi[1].start)
        energy /= roi_npix


    return energy


# def show_roi(stack, yslice, xslice):
def show_roi(stack, rois):

    """
    Display the roi over an image of the stack.
    :param stack: [nt x NY x NX x ncolorchannel]
    :param yslice: slice(start, end)
    :param xslice: slice(start, end)
    :return:
    """
    print(rois)
    zeros_mask = np.zeros_like(stack[0], np.uint8)
    for roi in rois:
        target_region = (slice(0, None), roi[0], roi[1], slice(0, None))
        zeros_mask[target_region[1:]] = 1

    plt.figure()
    plt.imshow(stack[20000] * zeros_mask)
    plt.xlabel('x');
    plt.ylabel('y');