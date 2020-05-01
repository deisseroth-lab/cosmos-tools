# COSMOS
Code to extract neuronal traces from a multi-focal microscope (COSMOS: Cortical Observation by Synchronous Multifocal Optical Sampling), and then to analyze these traces.

## Installation instructions
It is recommended that you use an [Anaconda](https://www.anaconda.com/distribution/)  environment for running this package. The following command line instructions assume Anaconda has been installed.

###### Download the repository 
`git clone https://github.com/izkula/cosmos-tools.git`

###### Install the environment
Navigate to the `cosmos-tools` folder created by git and then run:

`conda env create --file cosmos3requirements.yml -n cosmostools3`


###### Activate the new environment
`source activate cosmostools3`


###### Setup the cosmos package
`python setup.py develop`

###### Optionally setup ipywidgets package for jupyter notebook (only required for trace_merge_script.ipynb, and sometimes causes weird problems)
`conda install -c conda-forge ipywidgets`

(For some reason, [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html), which is necessary only for trace_merge_script.ipynb, appears to be required to be installed separately, after the environment has already been created. There have also been some issues when removing conda environments after ipywidgets has been installed, but remaking the environments appears to eventually work.)

###### Setup CNMF-E package for neural source extraction

The [CNMF-E implementation](https://github.com/zhoupc/CNMF_E) we use is written in MATLAB (a python implementation is in progress, and may be ready by the time you are reading this). 
This package uses the version of CNMF-E from June 27, 2018 (commit # ddb865832f52c81725859df8b5e032b4acb421e9).
A MATLAB license is thus required for a subset of this codebase (specifically, only for extracting sources from raw imaging data), all subsequent steps do not require MATLAB.
You can download the most up-to-date version of CNMF_E to your desired install location with

`git clone https://github.com/zhoupc/CNMF_E.git`

Then, open MATLAB, change to the installation directory, and run

`cnmfe_setup`

## Testing the code
From the top directory (in environment cosmos3) run:

`pytest`
or
`pytest --pdb`

If using Mac OSX, then because of an issue with matplotlib you instead need to do the following:

`conda install python.app`
then
`pythonw -m pytest`

## Additional installation notes

###### Note: There is a small possibility that due to a certain quirk in loading ome.tif files in import_raw_cosmos_data.py, you may need to use python2 for that script. It is possible the library call to load the files has now been properly upgraded for python3 and this is irrelevant. If it is necessary, you can likely just call python2 within the cosmos3 environment, or start a new environment (this has not been very well tested) with 
`conda env create --file install_stuff/cosmos2requirements.yml -n cosmos2 python=2.7`

`source activate cosmos2`

`python setup.py develop`


## Useful scripts (last updated 20200415)

### Importing raw data from COSMOS microscope:

1) Use `import_raw_cosmos_data.py`, following the directions at the top of that script. This crops the dual-lenslet ROIs, extracts timing information from the synchronization LED, uses CNMF_E to extract neural sources (requires MATLAB), and enables manual atlas alignment.

2) Then use the interactive jupyter notebook: `notebooks/processing_notebooks/trace_merge_script.ipynb` to align and merge traces from the different lenslets, quality control the extract sources, and save out an .h5 file containing merged traces that will be used for all further analyses of the dataset.

3) You can then run `notebooks/processing_notebooks/quickly_assess_merged_traces.ipynb` to quickly assess the sources and play with the traces. For general analysis, though, move the merged_traces.h5 file from processedData to ~/Dropbox/cosmos_data/[date]/[session_name]/, and add the relevant information about the session to cosmos.params.trace_analyze_params.py



### Importing intrinsic imaging movies for atlas alignment:

1) See `notebooks/processing_notebooks/intrinsic_imaging_alignment.ipynb`

2) After extracting phase map from step 1, you can manually overlay the phase map with the the image that contains vasculature. This alignment of PM/V1 boundary can then be used to precisely align already-imported COSMOS data, using `scripts/adjust_atlas.py`



### Importing two-photon data (i.e. for comparing visual stimulus orientation selectivity):

1) Process two-photon data (i.e. visual stimulus).
`scripts/batch_cnmf_2p.py`

2) Analyzing orientation selective visual grating stimulation.
`notebooks/primary_notebooks/fig_visual_stimulation_two_photon.ipynb`, 
`notebooks/primary_notebooks/fig_visual_stimulation_cosmos.ipynb`, 

### Analyzing COSMOS traces from lick-to-target task and generating figures:

#### Analyze merged traces.
`ipynb/primary_notebooks/trace_analyze_script_ik.ipynb`

#### Decode behavior from neural activity.
`notebooks/primary_notebooks/classification_analysis.ipynb`

#### Assign neural sources to task-related classes
`notebooks/primary_notebooks/task_class_assignment.ipynb`
`notebooks/primary_notebooks/fig_cluster_summary_with_mr2`

#### Trace summary figure
`notebooks/primary_notebooks/fig_trace_summary.ipynb`

#### Optics summary
`notebooks/primary_notebooks/fig_optics.ipynb`

#### Characterization of COSMOS vs. macroscope
`notebooks/primary_notebooks/fig_macroscope_comparison.ipynb`

#### Lick decoding
`notebooks/primary_notebooks/fig_classification_summary.ipynb`

#### VGAT optogenetic inhibition
`notebooks/primary_notebooks/vgat_inhibition_analysis.ipynb`

#### Unaveraged vs. trial-averaged correlation 
`notebooks/fig_cluster_summary_SINGLE_TRIAL` 

#### Optics simulations
`matlab/scripts/trace_analysis_spont.m` (for estimating background and signal photons) 
`matlab/SNR/dof_snr_simulations.m` (for plotting simulations)

## Reference:
**Please cite this paper when you use COSMOS in your research. Thanks!**

## License
Copyright 2020 Isaac Kauvar and Tim Machado


## General notes:
To run jupyter notebooks, on a remote computer, go to top directory and run:
`jupyter notebook --port=5558 --ip=* --no-browser`

See jupyter documentation to learn how to set up a password so that this is secure. 

