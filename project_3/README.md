# CSDS 335 Data Mining Project 3

The purpose of this project is to create and evaluate several classification models for two datasets, crop labels and their attributes in Ethiopia and India. The machine learning methods that we tested include Random Forest, Hist Gradient Boosting, Support Vector Machines, K-Nearest Neighbor, Logistic Regression, and Extra Trees. ((Discuss evaluation methods & metrics and maybe briefly results as well)) Please see our submission for a full report and presentation. 

The following are included:
- data_explore.ipynb
This is a Jupyter Notebook file containing all steps for analyzing and preparing our original datasets so they would match each other and for our understanding.
- machine_learning.ipynb
This is a Jupyter Notebook file contianing all machine learning train/test/evaluation steps during our experimentation. Cells can be run in order. 
- model_results.ipynb
This is a Jupyter Notebook file containing all final results/visualizaiton gathering. After we had a sense for our model performance from the previous file, this is where we made final comparisons for our report. Cells can be run in order.
- data/ethiopia_final.csv, data/india_final.csv, data/synthetic_final.csv
Datasets used to train/test ML models.
- data/crop_rec_ethiopia.csv, data/crop_rec_india.csv, data/crop_rec_synthetic.csv
CSV files saved during preprocessing steps, but not used in final model training/testing.

The following libraries are required: ((FINISH THIS LIST))
- pandas
- NumPy
- scikit-learn
- seaborn

Other notes: ((ADD LINKS))
- These datasets were found at the following links:
