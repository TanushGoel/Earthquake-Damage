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

1 represents low damage. 2 represents a medium amount of damage. 3 represents almost complete destruction. 

The totality of the data is available through the 2015 Nepal Earthquake Open Data Portal.
Following the 7.8 Mw Gorkha Earthquake on April 25, 2015, Nepal carried out a massive household survey using mobile technology to assess building damage in the earthquake-affected districts. Although the primary goal of this survey was to identify beneficiaries eligible for government assistance for housing reconstruction, it also collected other useful socio-economic information. In addition to housing reconstruction, this data serves a wide range of uses and users e.g. researchers, newly formed local governments, and citizens at large. The purpose of this portal is to open this data to the public.

More can be found out at [National Planning Commission Nepal](https://www.npc.gov.np/en) and [Earthquake Gorkha](https://en.wikipedia.org/wiki/April_2015_Nepal_earthquake).

1) geo_level_1_id, geo_level_2_id, geo_level_3_id (type: int): geographic region in which building exists, from largest (level 1) to most specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, level 3: 0-12567.
2) count_floors_pre_eq (type: int): number of floors in the building before the earthquake.
3) age (type: int): age of the building in years.
4) area_percentage (type: int): normalized area of the building footprint.
5) height_percentage (type: int): normalized height of the building footprint.
6) land_surface_condition (type: categorical): surface condition of the land where the building was built. Possible values: n, o, t.
7) foundation_type (type: categorical): type of foundation used while building. Possible values: h, i, r, u, w.
8) roof_type (type: categorical): type of roof used while building. Possible values: n, q, x.
9) ground_floor_type (type: categorical): type of the ground floor. Possible values: f, m, v, x, z.
10) other_floor_type (type: categorical): type of constructions used in higher than the ground floors (except of roof). Possible values: j, q, s, x.
11) position (type: categorical): position of the building. Possible values: j, o, s, t.
12) plan_configuration (type: categorical): building plan configuration. Possible values: a, c, d, f, m, n, o, q, s, u.
13) has_superstructure_adobe_mud (type: binary): flag variable that indicates if the superstructure was made of Adobe/Mud.
14) has_superstructure_mud_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Stone.
15) has_superstructure_stone_flag (type: binary): flag variable that indicates if the superstructure was made of Stone.
16) has_superstructure_cement_mortar_stone (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Stone.
17) has_superstructure_mud_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Mud Mortar - Brick.
18) has_superstructure_cement_mortar_brick (type: binary): flag variable that indicates if the superstructure was made of Cement Mortar - Brick.
19) has_superstructure_timber (type: binary): flag variable that indicates if the superstructure was made of Timber.
20) has_superstructure_bamboo (type: binary): flag variable that indicates if the superstructure was made of Bamboo.
21) has_superstructure_rc_non_engineered (type: binary): flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.
22) has_superstructure_rc_engineered (type: binary): flag variable that indicates if the superstructure was made of engineered reinforced concrete.
23) has_superstructure_other (type: binary): flag variable that indicates if the superstructure was made of any other material.
24) legal_ownership_status (type: categorical): legal ownership status of the land where building was built. Possible values: a, r, v, w.
25) count_families (type: int): number of families that live in the building.
26) has_secondary_use (type: binary): flag variable that indicates if the building was used for any secondary purpose.
27) has_secondary_use_agriculture (type: binary): flag variable that indicates if the building was used for agricultural purposes.
28) has_secondary_use_hotel (type: binary): flag variable that indicates if the building was used as a hotel.
29) has_secondary_use_rental (type: binary): flag variable that indicates if the building was used for rental purposes.
30) has_secondary_use_institution (type: binary): flag variable that indicates if the building was used as a location of any institution.
31) has_secondary_use_school (type: binary): flag variable that indicates if the building was used as a school.
32) has_secondary_use_industry (type: binary): flag variable that indicates if the building was used for industrial purposes.
33) has_secondary_use_health_post (type: binary): flag variable that indicates if the building was used as a health post.
34) has_secondary_use_gov_office (type: binary): flag variable that indicates if the building was used fas a government office.
35) has_secondary_use_use_police (type: binary): flag variable that indicates if the building was used as a police station.
36) has_secondary_use_other (type: binary): flag variable that indicates if the building was secondarily used for other purposes.
