function aspect_ratio = getAspectRatio(a)
%%%% Compute the aspect ratio of the object an image.
%%%% For example, the image may have the spatial fooprint of a neuron, 
%%%% and we want to ensure that only neurons with roughly symmetrical
%%%% footprints are accepted. 
%%%% Rather than fitting a gaussian, this function computes the radon
%%%% transform, which is the projection of the image across different
%%%% angles. Then the aspect ratio is computed based on the maximum and 
%%%% minimum width projection. This accounts for the possibility that the 
%%%% primary and secondary axes are tilted relative to vertical.
%%%% Input:
%%%%      a --- a 2D image (with, presumably, a single object in it)
%%%%  Output:
%%%%      aspect_ratio --- the ratio between the maximum and minimum widths
%%%%      of the object (potentially along a tilted set of axes). 

r = radon(a); 
R = r > 1; %%% Assumes images are scaled such that 'non-zero' signals have values greater than at least 1. 
Rwidths = sum(R, 1);
aspect_ratio = max(Rwidths)/min(Rwidths);
    