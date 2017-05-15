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

#### Very imbalanced dataset 

A first observation is that the dataset is very imbalanced towards people who earn less than 50k/year (93% of the dataset !).  It is necessary to take this into account in machine learning models we build, particularly when tuning the parameters (`StratifiedKFold`). 

#### distinct profiles for people with low income / people with higher income
Average age for people with higher income is 46, whereas it is 33 years old for people with lower income.
The top category for `education` is Children among the lower -50k category, and Bachelor degree among the +50k one. This latter category also has a majority of `married` people, while the other is mainly formed of `never married` people. The 50k+ category is mostly male while the 50k- is mostly female, and both are in majority white.
The +50k category works in majority in `Manufacturing - durable goods`, with the `Executive admin and managerial` status. 

### Feature Engineering : selecting the most relevant features

#### Discarding useless variables





### Classification 

### Further steps

#### 'Automated' feature engineering
