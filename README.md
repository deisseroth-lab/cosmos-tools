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

Additionally, the CNMF-E implementation (https://github.com/zhoupc/CNMF_E) we use is written in MATLAB (a python implementation is in progress). Thus, a MATLAB license is required for parts of this codebase.

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

##### -Process two-photon data (i.e. visual stimulus).
`batch_cnmf_2p.py`

##### -Crop dual lenslet ROIs, deal with timing LED, perform cnmfe, and align atlas. This script should be run using cosmos2 environment.
`dual_lenslet_crop.py`
###### For the cell extraction in this script to work, you also need to install CNMF_E, found here: https://github.com/zhoupc/CNMF_E
###### You can do this, for example by going to the desired install location,
`git clone https://github.com/zhoupc/CNMF_E.git`
###### Opening matlab, changing to that directory, and
`cnmfe_setup`

##### -Intrinsic imaging based alignment.
`ipynb/intrinsic_imaging_alignment.ipynb`

##### -Adjusting atlas alignment to match intrinsic imaging.
`scripts/adjust_atlas.py`

##### -Analyzing orientation selective visual grating stimulation.
`ipynb/COSMOS Visual Analysis.ipynb`, 
`ipynb/COSMOS Visual Stimulation Figure.ipynb`

##### -Clean up extracted traces, align and merge traces from the different lenslets, and save out an .h5 file containing merged traces, and aligned atlas.
`ipynb/trace_merge_script.ipynb`


Note: in some instances, you may need to setup ipython widgets

`conda install -n base -c conda-forge widgetsnbextension`

`conda install -n cosmos3 -c conda-forge ipywidgets`

`jupyter nbextension enable --py --sys-prefix widgetsnbextension`

See: https://ipywidgets.readthedocs.io/en/stable/user_install.html

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
