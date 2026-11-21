# CSDS 335 Data Mining Project 3

The purpose of this project is to create and evaluate several classification models for two datasets, crop labels and their attributes in Ethiopia and India. The machine learning methods that we tested include Random Forest, Hist Gradient Boosting, Support Vector Machines, K-Nearest Neighbor, Logistic Regression, and Extra Trees. The seven input features used for prediction are nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, and rainfall.

We evaluated the models using an 80/20 stratified train/test split, cross validation, and the metrics accuracy, macro F1, and weighted F1. Macro F1 was helpful for the Ethiopian dataset because of the class imbalance. We also tried including class weighting and hyperparameter tuning to improve Random Forest performance on the Ethiopian dataset. Overall, tree-based ensemble methods achieved the strongest results. The project report gives a more detail description of methodology, background, and results.


The following are included:
- data_explore.ipynb
This is a Jupyter Notebook file containing all steps for analyzing and preparing our original datasets so they would match each other and for our understanding. This includes matching the Ethiopian and Indian feature sets, processing climate variables, and examining crop distributions.
- machine_learning.ipynb
This is a Jupyter Notebook file contianing all machine learning train/test/evaluation steps during our experimentation. Cells can be run in order. 
- model_results.ipynb
This is a Jupyter Notebook file containing all final results/visualizaiton gathering. After we had a sense for our model performance from the previous file, this is where we made final comparisons for our report. Cells can be run in order.
- data/ethiopia_final.csv, data/india_final.csv, data/synthetic_final.csv
Datasets used to train/test ML models.
- data/crop_rec_ethiopia.csv, data/crop_rec_india.csv, data/crop_rec_synthetic.csv
CSV files saved during preprocessing steps, but not used in final model training/testing.

The following libraries are required: 
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- jupyter

run this first to install:
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

**Running**
1. Download code
2. Make sure directories match this and you have all of the csvs
3. Install the libraries, see above
4. Start jupyter notebook in a terminal: jupyter notebook
5. Open the notebooks and run their cells in order

- order to fun files: 
data_explore.ipynb 
machine_learning.ipynb 
model_results.ipynb 

Other notes: 
- These datasets were found at the following links:
    - Ethiopia: https://data.mendeley.com/datasets/8v757rr4st/1/files/98242fd3-1912-4a59-ab26-23d97b454218
    - India: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset