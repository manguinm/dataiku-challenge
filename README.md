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
It makes sense to define an age threshold under which we can reasonably assume people are too young to earn 50k+ per year. From what we learned from the training set, we define this threshold as age_min = 15 years old and put this young subset aside of the study. It helps rebalance a bit the dataset : in the adult one, there   

#### Distinct profiles for people with low income / people with higher income
- Age : Average age for people with higher income is 46, whereas it is 33 years old for people with lower income.
- Education : The top category for `education` is Children among the lower -50k category, and Bachelor degree among the +50k one. This latter category also has a majority of `married` people, while the other is mainly formed of `never married` people. The 50k+ category is mostly male while the 50k- is mostly female, and both are in majority white.
- Working class : The +50k category works in majority in `Manufacturing - durable goods`, with the `Executive admin and managerial` status. The -50k has a majority of people who have never worked.


### Feature Engineering : selecting the most relevant features

#### Discarding useless variables

First, I removed the **instance-weight** variable as the purpose of this variable is to readjust the sample to be representative of the whole population, and should not be used for classification. 

Trees methods in scikit-learn provide a useful `feature_importances` metrics for getting information on the relative importance of each feature, based on how 'high' a feature is used to split at a node. By using an `ExtraTree Classifier` (I one-hot encoded the categorical variables before), I discarded the following variables, which were the least important in the tree:
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

The metrics we are most interested in, in this case of unbalanced dataset 


### Further steps

- Refining features, combining them ('handcrafted' ratios, Principal component analysis...)

#### 'Automated' feature engineering
