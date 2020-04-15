%%% MASTER SCRIPT FOR COSMOS microscope design. 

%%% 20171031 - For ray optics simulation of different depth of field
%%% approaches.
SNR/dof_snr_simulations.m


%%% 20171031 - For analyzing SNR across the window with different
%%% techniques. (originally used SNR/online_snr.m). This is inferior to
%%% just running the videos through CNMF-e and looking at the resultant
%%% traces and neuron counts though...
SNR/compare_snr.m


%%% 20171120 - For comparing microscope designs with the pinhole psfs.
PSFs/analyzePSFs.m


%%% 20171121 - For analyzing cropped rois
scripts/crop_ROIs.m
scripts/batch_cnmfe.m
scripts/trace_analysis.m %%% Note: for this, right now you need to move the final file with traces to Dropbox (cosmos_data/date/...)
scripts/light_collection_comparison.m %%% For plotting mean intensity of ROIs, comparing light throughput of different designs.
scripts/trace_analysis_spont.m %%% For comparing cnmf-e results of different optical designs, recorded without bpod. Used for figures made over winter break 2017. 

%%% For making a video with tongue and neural co-registered, in python:
cosmos/analysis_scripts/align_tongue_neural_vid.py
analysis_scripts/plot_licks.py

%%% Useful for behavioral analysis?
COSMOSOrganizeTrialTraceData

%%% For generating atlas that is read into python
atlas/process_atlas_script.m