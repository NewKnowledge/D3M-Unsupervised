testing change
# D3MUnsupervised Code

## Hdbscan.py (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
Wrapper of the DBSCAN and HDBSCAN clustering algorithms into D3M infrastructure. 

## Storc.py 
Wrapper of the KMeans clustering algorithm into D3M infrastructure. 
The base implementation of KMeans can be found here: https://github.com/NewKnowledge/sloth

## timeseries_formatter.py
Formatting primitive which converts nested dataframe of filenames with individual time series into a single dataframe of time series observations. 
Allows the KMeans and Hdbscan primitives to be applied to time series clustering problems. 

# pipelines

## Hdbscan_pipeline_basic.py
Basic Hdbscan pipeline whiches uses Gradient boosting classifier.

## Hdbscan_pipeline_best.py
Best Hdbscan pipeline which uses an ensemble of Random Forests and does some data cleaning (imputation) before clustering.

## Sloth_pipeline_SEMI_1040_sylva_prior.py
Sloth pipeline applied to a specific semi-supervised dataset. Number of clusters hyperparameter in this pipeline optimized for this dataset.

# scripts

## docker_setup.sh
Shell script to update your fork of the primitives repository to the latest program version.

## Hdbscan_testing.sh
Shell script to test Hdbscan pipeline on a list of datasets that saves pipelines and writes scoring and timing information to a file.

## image_and_dataset_setup.sh
Shell script to download datasets (files managed and not managed by LFI) and pull and run latest version of program image. 
