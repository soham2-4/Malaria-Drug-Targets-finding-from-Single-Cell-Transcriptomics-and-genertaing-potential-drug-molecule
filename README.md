# Computational approach for decoding Malaria Drug targets from Single-Cell Transcriptomics and finding potential drug molecules.

Datasets: Three datasets are given in the Datasets folder. 

Data preprocessing: The datasets given in this repository are preprocessed. We have collected raw datasets from three papers(details in the paper). Then, we preprocessed these datasets to get a meaningful representation of our work.

Feature selection using mutual-information-based feature reduction technique:  Code folder has all the python files.'without_featureselection.py' can be used to get the classification results before feature selection. 'MI_feature_reduction.py' can be used to get the classification result after selecting features using Mutual information-based feature selection. 'random_feature.py' selects random features from the dataset and gets classification results. classification_results.ipynb plots bar charts of comparisons between different classifiers. 

Protein-protein interaction network analysis: After feature selection, we did Protein-protein interaction network analysis with these features. All the results can be found in 1st dataset. 2nd dataset and 3rd dataset folders.

Crucial proteins:

Suggested drug molecules using Targetdiff:

ADMET and druglikeliness:



