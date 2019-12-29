# Earthquake-Damage
-DrivenData competition - Richter's Predictor: Modeling Earthquake Damage - predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal based on aspects of the building location and construction

-A categorical boosting (catboost) classifier that predicts the damage grade of a building as either 1, 2, or 3 given multiple categorical and binary features of the building

-There are many binary features in the datasets of this competition that would not be great inputs for the decision trees of the catboost model to train on. 
-Certain binary features such as "has_superstructure_adobe_mud" or "has_superstructure_mud_mortar_stone" or "has_superstructure_cement_mortar_brick" could be turned into a single categorical "superstructures" feature.
-Similarly, binary features such as "has_secondary_use_agriculture" or "has_secondary_use_industry" or "has_secondary_use_institution" can be turned into a single categorical "secondary" feature.

-The main difference between the 1st and 2nd notebooks is just how I tried to turn the many binary features into categorical features.
The first notebook essentially checks if the building has the given superstructure or secondary feature (in which case the value of the feature is 1), and if it does have it, it will add the feature name to the end of a string that is later factorized into numerical values.

For example,
    -has_superstructure_adobe_mud: 1
    -has_superstructure_mud_mortar_stone: 0
    -has_superstructure_cement_mortar_brick: 1

-This specific combination of superstructures may have a factorized numerical value of say 9 (the factorized value depends on the strings factorized before it - but it simply tries to give different strings unique values)
superstructures: "adobe_mud_cement_mortar_brick" --> 9

-At the end, every unique combination of having certain superstructures or secondaries will have a different number.


The second notebook simply appends a certain number to the end of a string, turning it into an integer at the end
