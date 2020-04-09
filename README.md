# COSMOS

Herein lies code to extract neural traces from a multi-focal microscope, and then to analyze these traces.

## Setting up environment 
Two environments are necessary (one is python2 and one is python3), due to a quirk in loading certain types of tif stacks in dual_lenslet_crop.py, for which python2 is necessary. Everything else can safely use python3.

`conda env create --file cosmos3requirements.yml -n cosmos3 python=3.5`

`conda env create --file cosmos2requirements.yml -n cosmos2 python=2.7`

###### Then do
`source activate cosmos3`

`python setup.py develop`

###### and
`source activate cosmos2`

`python setup.py develop`

Additionally, the CNMF-E implementation (https://github.com/zhoupc/CNMF_E) we use is written in MATLAB (a python implementation is in progress, and potentially will be ready by the time you are reading this). Thus, a MATLAB license is required for a subset of this codebase (specifically, it is only necessary for extracting sources from raw imaging data), all subsequent steps do not require MATLAB.
You can download the most up-to-date version of CNMF_E by changing directories to the desired install location, and then running

`git clone https://github.com/zhoupc/CNMF_E.git`

Then, open MATLAB, change to the installation directory, and run

`cnmfe_setup`


## -Testing the code. 
From the top directory (in environment cosmos3) run:

`pytest`
or
`pytest --pdb`

If using Mac OSX, then because of an issue with matplotlib you instead need to do the following:

`conda install python.app`
then
`pythonw -m pytest`

## Useful scripts (Last updated 20191201.)

#### Importing raw data from COSMOS microscope:

1) Use `import_raw_cosmos_data.py`, following the directions at the top of that script. This crops the dual-lenslet ROIs, extracts timing information from the synchronization LED, uses CNMF_E to extract neural sources (requires MATLAB), and enables manual atlas alignment.

2) Then use the interactive jupyter notebook: `ipynb.trace_merge_script.ipynb` to align and merge traces from the different lenslets, quality control the extract sources, and save out an .h5 file containing merged traces that will be used for all further analyses of the dataset.

Note: in some instances, you may need to setup ipython widgets

`conda install -n base -c conda-forge widgetsnbextension`

`conda install -n cosmos3 -c conda-forge ipywidgets`

`jupyter nbextension enable --py --sys-prefix widgetsnbextension`

See: https://ipywidgets.readthedocs.io/en/stable/user_install.html


#### Importing intrinsic imaging movies for atlas alignment.

1) See `ipynb/intrinsic_imaging_alignment.ipynb`

2) After extracting phase map from step 1, you can manually overlay the phase map with the the image that contains vasculature. This alignment of PM/V1 boundary can then be used to precisely align already-imported COSMOS data, using `scripts/adjust_atlas.py`



#### Importing two-photon data (i.e. for comparing visual stimulus orientation selectivity)

##### -Process two-photon data (i.e. visual stimulus).
`batch_cnmf_2p.py`

##### -Analyzing orientation selective visual grating stimulation.
`ipynb/COSMOS Visual Analysis.ipynb`, 
`ipynb/COSMOS Visual Stimulation Figure.ipynb`

##### -Clean up extracted traces, align and merge traces from the different lenslets, and save out an .h5 file containing merged traces, and aligned atlas.
`ipynb/trace_merge_script.ipynb`

##### -Analyze merged traces.
`ipynb/trace_analyze_script_ik.ipynb`

##### -Decode behavior from neural activity.
`ipynb/classification_analysis.ipynb`

##### - Clustering synchronously recorded sources
`ipynb/cluster_analysis.ipynb`

##### -Generalized linear model for predicting neural activity from behavior.
`ipynb/glm_development.ipynb`

##### Analyze behavior videos
`ipynb/extract_video_regressors.ipynb`

### Figure notebooks
#### Trace summary figure
`ipynb/fig_trace_summary.ipynb`

#### Optics summary
`ipynb/fig_optics.ipynb`

#### Characterization of COSMOS vs. macroscope
`ipynb/fig_macroscope_comparison.ipynb`

#### Lick decoding
`ipynb/fig_classificiation_summary.ipynb`

#### VGAT inhibition
`ipynb/vgat_inhibition_analysis.ipynb`

#### Orientation selectivity
`ipynb/fig_visual_stimulation_cosmos.ipynb`
`ipynb/fig_visual_stimulation_two_photon.ipynb`

#### Task-classification
`ipynb/task_class_assignment` (for mean-correlation based)
`ipynb/glm_class_assignment` (for glm based)
`ipynb/fig_cluster_summary_with_mr2` (for mean-correlation based)
`ipynb/fig_cluster_summary_with_glm` (for glm based)

#### Local clustering
`ipynb/fig_cluster_summary_SINGLE_TRIAL` (for single-trial vs trial-averaged clustering)

#### Optics simulations
`matlab/scripts/trace_analysis_spont.m` (for estimating background and signal photons)
`matlab/SNR/dof_snr_simulations.m` (for plotting simulations)




## General notes:
To run ipynb, on a remote computer, go to top directory and run:
ipython notebook --port=5558 --ip=* --no-browser
