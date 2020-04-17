#!/usr/bin/python
"""
Usage:
from VideoSlider import vslide
vslide(vid)

For example:
import numpy as np
from VideoSlider import vslide

if __name__ == "__main__":

    N = 1000
    vid = np.zeros((10, 10, N))
    print(np.shape(vid)[2])
    for i in range(N):
        vid[:,:,i] = i

    vslide(vid)
"""

import Tkinter
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
import numpy as np


def vslide(vid, low=None, high=None, cmap=None):
    """
    Wrapper function that calls initiates video_slider GUI
    Usage: vslide(vid, low, high, cmap)

    :param vid: a 3D numpy array (X, Y, T)
    :param low: lower color limit
    :param high: uppder color limit
    :param cmap: colormap
    """
    VideoSlider(vid, low, high, cmap).mainloop()


class VideoSlider(Tkinter.Tk):
    """
    GUI for stepping through a video.
    Usage: from VideoSlider import video_slider
           video_slider(vid).mainloop()

    Where vid is a 3D numpy array (X, Y, T).
    """
    def __init__(self, vid, low=None, high=None, cmap=None):
        self.parent = None
        Tkinter.Tk.__init__(self, self.parent)
        self.title('video_slider')
        self.__vid = vid
        self.__low = low
        self.__high = high
        self.__cmap = cmap
        self.__T = np.shape(vid)[2]
        self.initialize()

    def initialize(self):
        if self.__low is None:
            self.__low = np.min(self.__vid)

        if self.__high is None:
            self.__high = np.max(self.__vid)

        if self.__cmap is None:
            self.__cmap = "Greys"

        self.canvasFig = plt.figure(1)
        Fig = matplotlib.figure.Figure(figsize=(10, 10), dpi=100)
        a = Fig.add_subplot(111)
        self.a = a
        # im = a.imshow(2*np.ones((10,10)), clim=[-4, 4])
        self.im = a.imshow(self.__vid[:, :, 3],
                           clim=[self.__low, self.__high],
                           cmap=self.__cmap)
        Fig.colorbar(self.im, ax=a)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            Fig, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(
            side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        self.canvas._tkcanvas.pack(
            side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)

        self.entry = Tkinter.Scale(self, from_=0, to=self.__T-1,
                                   command=self.refreshFigure,
                                   length=1000,
                                   width=40,
                                   orient=Tkinter.HORIZONTAL).pack(
                                       side=Tkinter.TOP)  # create entry widget

        self.resizable(True, True)
        self.update()

    def refreshFigure(self, val):
        self.im.set_data(self._VideoSlider__vid[:, :, int(val)])
        self.a.set_title(val)
        self.canvas.draw()
        pass


if __name__ == "__main__":
    N = 10
    vid = np.zeros((10, 10, N))
    print(np.shape(vid)[2])
    for i in range(N):
        vid[:, :, i] = i
    VideoSlider(vid).mainloop()
