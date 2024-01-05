# Computational approach for decoding Malaria Drug targets from Single-Cell Transcriptomics and finding potential drug molecules.

Datasets: Three datasets are given in the Datasets folder. 

Data preprocessing: The datasets given in this repository are preprocessed. We have collected raw datasets from three papers(details in the paper). Then, we preprocessed these datasets to get a meaningful representation of our work.

Feature selection using mutual-information-based feature reduction technique:  Code folder has all the python files.'without_featureselection.py' can be used to get the classification results before feature selection. 'MI_feature_reduction.py' can be used to get the classification result after selecting features using Mutual information-based feature selection. 'random_feature.py' selects random features from the dataset and gets classification results. classification_results.ipynb plots bar charts of comparisons between different classifiers. All the selected features can be found in 1st dataset. 2nd dataset and 3rd dataset folders.

Protein-protein interaction network analysis: After feature selection, we did Protein-protein interaction network analysis with these features. All the results can be found in 1st dataset. 2nd dataset and 3rd dataset folders. With these selected features, we did enrichment analysis of biological functions.

Crucial proteins: After doing Protein-protein interaction network analysis, we selected crucial proteins from these networks with high degree and betweenness centrality and found their importance for plasmodium survival. 

Suggested drug molecules using Targetdiff: Next, we found strong binding sites of crucial proteins and generated drug molecules using targetdiff(a generated deep learning framework).

ADMET and drug-likeliness: In this step, we analyzed ADMET and drug-likeliness properties of all the generated drug molecules and found some drug molecules that can work as potential drug molecules.



