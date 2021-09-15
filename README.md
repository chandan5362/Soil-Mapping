
## Introduction
This project is a part of semester long project. It is meant to be completed in two semeter( 6th and 7th). So, Below is the progress done so far. In the next semester, I will be working further on the spatial result improvement and integration of the same into a cross-platform  mobile app.

Soil nutrient details can help farmers immensely in choosing the appropriate crop and using the necessary fertilizers. However, collecting soil nutrient data is very cumbersome and time-consuming. With deteriorating soil health and crop yield, it is essential to come up with better ways to help farmers. Limited and relatively sparse values of soil concentrations can be used to interpolate, estimate and predict nutrient information at all geographic points in the vicinity.
Soil nutrient data as provided in the Soil Health Cards by the DAC, Govt. of India have been used as the known points in our approach. This dataset contains concentrations of Nitrogen, Potassium, Phosphorus and Organic Compounds at about 25k geolocations. Fig(1) shows their distribution of scaled Nitrogen(N) across Maharashtra, India.

![img](https://lh5.googleusercontent.com/WG1Z5c6iSrSHZtBX1toZGsqIC6ZEc7M6L_If28GOwSOY15M4WIXzPLvlavUhNYISLGv9OahaVp-kOjJgZRg15RAIa6e99WgxXe9N8b4P70XW6EI0G2YMc_TLfIoBhlKNE7zvUgSu=s0)

## Methodology

Nitrogen values from the data set were used for interpolation and prediction. Here, we used 80-20 training and test split. The values of Nitrogen concentration were interpolated at about 60,000 points from the known 25,000 points in the dataset. These were used for spatial mapping.
* Inverse distance weighting
* Ordinary Krigging
* KNN Regression

Our main method was to estimate the nutrients at unknown location using Neural Process

### Neural Process
Neural process(Garnelo et al.,2018) is a kind of regression by learning to map a context set of observed input and output pairs to a distribution over regression functions. Each function models the distribution of an output given an input variable conditioned on the context. It can learn a wide family of conditional distributions. Since the NP has a fundamental drawback of underfitting, we have used Attentive Neural Process( Garnelo et al.,2018) allowing each input to attend to the relevant context points of prediction. Since our dataset is a real world dataset and it suffers from non-stationarity, we will be including it in further evaluation. 

Model

![img](https://lh4.googleusercontent.com/bijN1XFX1peFnOaxxbHN1ZwQtAyyf2INs8wR6vjC68zEjfo8QJAdfcaIY_fI5SCCJDZ6dwFSvTyCQXhY1EEnLqJvOv5IubCRi2emnewcZwqdy5qPh6AcrhTAvQmocdJ1HDOPmgpT=s0)



### Results

Once the training is done, we sample all the points within the convex hull and perform inference over all these sampled points. In our case, we sampled 62k points including the training point

![img](https://lh4.googleusercontent.com/izL8ug0Ci2sFmaTN0DZQwXdDsZkVj286Q9GLNgq_N4zcw7LSvrq8GtKPxz7-YTo1-FepVFX_bqdqdpQIuuvKmA50aN-QipSyKo_S9nEHydWdh4VfHDbttXZoV9KVEfkr13JW6ouJ=s0)

### Interpolation

![img](https://lh3.googleusercontent.com/C48MFC0IxEiZP6gzsrbqvneLMoqwD0cQI0P9NYNY7MYzj7NP2dDPkawuJYJ2h00aIjdvbG5u5RqXI0gNepddCRXSCVaeSN15h_zoxVelzbbVsPOdd1Be0iwuHPyBUrkmu_66y2Ci=s0)

![img](https://lh3.googleusercontent.com/2CNm_hmtiewIjggJbQTP7Z1ltieXeo_7eFK_V6qYKdykPE7Dbn4XH-8IdS8MROh6ODMOW2sC-IKT5LCpTy2ZLuVyt2yEfoJVw_a6-q0YlCxh7_CN9YhV3_7tVwbItTYQBYN6CATa=s0)



it is clear from the figure that most of the interpolated points are in the positive standard deviation as the left figure is denser than the right one.













