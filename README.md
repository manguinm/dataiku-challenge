# Dataiku challenge : Predicting income level from US Census Data

My submission for Dataiku's challenge

## Dataset description

We are given US Census data records for roughly 200,000 individuals. For each record, we have 42 socioeconomic variables, among which we are asked to predict the last, which indicates whether the individual earns more or less than 50,000$ per year.

### Missing values 
There are not blank cells, but the data is skewed in another way. 
Because of the variety of the variables and the large range of profiles, many variables can not have a *proper* value for individuals (e.g. number of work days in a year for children), this is encoded by the **Not in universe** value and its variants. 

Furthermore, the `migration_code_something` variables have a lot of unknown values. This is why we choose to discard them in further steps.

## Method

### Tools

- [pandas](http://pandas.pydata.org/) for data manipulation
- [seaborn](http://seaborn.pydata.org/index.html) and matplotlib for visualization
- `scikit-learn` for machine learning tools

### Exploratory Data Analysis : key statistics

To get a better understanding of the dataset and the repartition of the variables, I made a series of univariate and bivariate plots, a correlation matrix among continuous variables, and some co-occurence matrices for categorical labels. The main learnings I got are described below:

#### Very imbalanced dataset 

A first observation is that the dataset is very imbalanced towards people who earn less than 50k/year (93% of the dataset !).  It is necessary to take this into account in machine learning models we build, particularly when tuning the parameters (e.g. `StratifiedKFold` and a good choice of performance metrics broken down by category). 

#### Choice : working on an *adult* subset of the dataset
It makes sense to define an age threshold under which we can reasonably assume people are too young to earn 50k+ per year. From what we learned from the training set, we define this threshold as age_min = 15 years old and put this young subset aside of the study. It helps rebalance a tiny bit the dataset (92% of the adult dataset are under 50k, while 94% in the original dataset). 

#### Distinct profiles for people with low income / people with higher income
- Age : Average age for people with higher income is 46, whereas it is 33 years old for people with lower income.
- Education : The top category for `education` is Children among the lower -50k category, and Bachelor degree among the +50k one. This latter category also has a majority of `married` people, while the other is mainly formed of `never married` people. The 50k+ category is mostly male while the 50k- is mostly female, and both are in majority white.
- Working class : The +50k category works in majority in `Manufacturing - durable goods`, with the `Executive admin and managerial` status. The -50k has a majority of people who have never worked.


### Feature Engineering : selecting the most relevant features

#### Discarding useless variables

First, I removed the **instance-weight** variable as the purpose of this variable is to readjust the sample to be representative of the whole population, and should not be used for classification. 
I ralso removed variables that provided redundant info, such as detailed_industry_code.

Trees methods in scikit-learn provide a useful `feature_importances` metrics for getting information on the relative importance of each feature, based on how 'high' a feature is used to split at a node. By using an `ExtraTree Classifier` (I one-hot encoded the categorical variables before), I discarded the least important variables in the tree. To sum up, I dropped all the following variables:

- detailed industry recode and detailed occupation recode
- migration-related variables
- year (assuming year does not play a role here, as it is only on a 2-year period)
- country of birth father and mother
- detailed household and family stat
- state of previous residence


This `ExtraTree` also provided me with useful insights on the most important features, which are mainly age, education level and work-oriented variables. Interestingly, financial variables like capital gains, dividends and capital losses play a huge role. In further investigation, one could merge these three into a financial_health indicator for instance.

I also simplified the **education** variable into a simpler variable with fewer categories. Which led me to take the following reduced list of features:
LIST

I could have shortened the list a lot more. But I did not want to lose any potential information at this stage. 

### Classification 
I tried the following classification methods:
- Logistic Regression 
- Decision Tree (I plotted a simple decision tree with a small depth for a nice viz)
- Random Forest

For each classifier, I got the training performance by cross-validating on a stratified KFold = 5 folds, optimizing on the F-1 score. 
If hyperparameter tuning needed, I would also use cross-validation with stratified folds, optimizing on the F-1 score.
I carefully printed the `classification report` on the test set to get details on performance metrics.

#### Key metrics: 
The metrics we are most interested in, in this case of unbalanced dataset, are precision and recall for the 50k+ category : we want to make sure we flag accurately all people in this category and we do not want to falsely flag someone as in the 50k+ category while he is not.  

#### Summary of the results

Results on the adults dataset
* N.B : we achieve 100% accuracy in flagging people under < 15 in the < 50k category.*

| Classifier | Logistic Regression | Decision Tree | Random Forest
--- | --- | --- | --- 
| - 50k precision | 0.98 |  0.98 | 0.95
| - 50k recall | 0.83 | 0.80 | 0.98
| < 50k F-1 score |0.90 | 0.90 | 0.97
| 50k +precision | 0.31 | 0.28  | 0.70
| 50k + recall | 0.86 | 0.85 | 0.39
| 50k + F-1 score | 0.46 | 0.42 | 0.50



The classification methods have a nice class_weight parameter that I defined as `balanced` in order to address the imbalanced dataset problem. 

Overall, with the Decision Tree and the Logistic Regression, we achieve a good recall for the 50k+ category ! But the precision is  bad. Which means we manage to flag most of the 50k+ people (few false negatives), but we also flag wrongly many people in the 50k+ category (many false positives). 
The Random Forest yields the best resuls (the best F1-score for the 50k+ category), even though it is computationnaly costly in this high-dimensional features matrix. Interestingly, with this RF, we achieve a better precision than a recall.



### Further steps

Because of time constraints, I had to end this work at this point, but there are definitely more steps to explore to tackle this classification problem. From most important to least important in my opinion, here are the further techniques I would like to look at : 

- use **oversampling techniques** to rebalance the training set and help classifiers learn more the 50k+ category patterns (the [imbalanced_learn](https://github.com/scikit-learn-contrib/imbalanced-learn) module in Python seems a good tool)
- Refining features, combining them ('handcrafted' ratios, Principal component analysis...)
- Instead of using a classification approach, use an anomaly detection one (cluster analysis-based for instance)



