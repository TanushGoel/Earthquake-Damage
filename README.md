# Earthquake-Damage
DrivenData competition - [Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) - predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal based on aspects of the building location and construction

A categorical boosting (catboost) classifier that predicts the damage grade of a building as either 1, 2, or 3 given multiple categorical and binary features of the building.
A random forest was also attempted, but achieved a lower accuracy than the catboost model. An ensemble of both models also resulted in a decreased accuracy.

## Feature Handling - Binary --> Categorical

There are many binary features in the datasets of this competition that would not be great inputs for the decision trees of the catboost model or random forest to train on. 


Certain binary features such as *"has_superstructure_adobe_mud"* or *"has_superstructure_mud_mortar_stone"* or *"has_superstructure_cement_mortar_brick"* could be turned into a single categorical *"superstructures"* feature.


Similarly, binary features such as *"has_secondary_use_agriculture"* or *"has_secondary_use_industry"* or *"has_secondary_use_institution"* can be turned into a single categorical *"secondary"* feature.

## Notebooks
The main difference between the 1st and 2nd notebooks is just how I tried to turn the many binary features into categorical features.

### Notebook #1
The first notebook essentially checks if the building has the given superstructure or secondary feature (in which case the value of the feature is 1), and if it does have it, it will add the feature name to the end of a string that is later factorized into numerical values.

For example,

    - has_superstructure_adobe_mud: 1
    - has_superstructure_mud_mortar_stone: 0
    - has_superstructure_cement_mortar_brick: 1

This specific combination of superstructures may have a factorized numerical value of say 9 (the factorized value depends on the strings factorized before it - but it simply tries to give different strings unique values)

    - superstructures: "adobe_mud_cement_mortar_brick" --> 9

At the end, every unique combination of having certain superstructures or secondaries will have a different number.

### Notebook #2
The second notebook simply appends a certain number to the end of a string, turning it into an integer at the end. 

For example,

    - has_superstructure_adobe_mud: 1
    - has_superstructure_mud_mortar_stone: 0
    - has_superstructure_cement_mortar_brick: 1

I would make the adobe_mud superstructure append "1" to a string, mud_mortar_stone append "2", and have cement_mortar_brick append "3" if it has the superstructure - then turn the string into a numerical value.

You can see that if a building has more superstructures, the number would be much larger. On top of this, I guarantee that even if a certain combination seen in the testing set is not even seen in the training set, the model would still be able to gain a sense of what type of superstructure(s) it has. This method seems like a much more organized approach for turning binary features to categorical, instead of the original one where I gave random distinct numbers for every combination seen in the training set. 

    - superstructures: "13" -- > 13
  
This method greatly increased the feature score determined through SelectKBest from sklearn's feature selection package. 

Oversampling was also tried in the second notebook but had a deproved testing accuracy. 

### Libraries
- [Pandas](https://github.com/pandas-dev/pandas)
- [Numpy](https://github.com/numpy/numpy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Seaborn](https://github.com/mwaskom/seaborn)
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
- [Catboost](https://github.com/catboost/catboost)

### Data
[all_datasets](https://www.drivendata.org/competitions/57/nepal-earthquake/data/)
