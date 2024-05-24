# AirBnB Ratings - Data Modeling and Predictive Analytics

## Overview
This project is part of the Data Mining and Predictive Analytics course (BUDT758T) led by Professor Jessica Clark. I focused on predicting the perfect rating score of Airbnb listings using various data mining techniques and models.

## Business Understanding
### Business Cases
1. **Airbnb Hosts**: Optimize listings to achieve higher ratings.
2. **Vacation Rental Management Businesses**: Improve property management strategies.
3. **Potential Investors**: Make informed investment decisions.
4. **Airbnb Competitors**: Benchmark and enhance their own platforms.

### Value Generated
- **Listing Optimization**: Enhancing guest experiences to boost ratings.
- **Risk Assessment**: Informed investment decisions to mitigate risks.

## Data Understanding and Preparation
### Model Features
We utilized a wide range of features, including:
- **Original Features**: Availability, bathrooms, beds, cancellation policy, etc.
- **Engineered Features**: Ratios like price-to-accommodates, feature transformations, etc.

### Feature Insights
1. **Impact Analysis**: Using table and summary functions to understand feature impact.

   ![image](https://github.com/shahnishank/AirBnB_Ratings-Predictive_Analytics/assets/108402877/49f7875f-d29b-416e-b318-ab25968143a5)

3. **Correlation Matrix**: Identifying and avoiding redundant features.

5. **P-Value Analysis**: Determining the significance of each feature.

   <img width="314" alt="image" src="https://github.com/shahnishank/AirBnB_Ratings-Predictive_Analytics/assets/108402877/7d67e4ad-5d87-458c-8e76-da881dbce6ae">

7. **Feature Gain**: Quantifying the contribution of each feature to the model's predictive power.

   <img width="425" alt="image" src="https://github.com/shahnishank/AirBnB_Ratings-Predictive_Analytics/assets/108402877/bcc74cf4-91e2-4971-be09-94c6f21a88b8">


## Feature Selection 

# Feature Table

| ID  | Feature Name                         | Brief Description                                              | R Code Line Numbers |
|-----|--------------------------------------|----------------------------------------------------------------|---------------------|
| 1   | availability_30                      | Original feature from dataset                                  | 171                 |
| 2   | availability_365                     | Original feature from dataset                                  | 174                 |
| 3   | bathrooms                            | Original feature from dataset                                  | 167                 |
| 4   | beds                                 | Original feature from dataset                                  | 166                 |
| 5   | cancellation_policy                  | Original feature from dataset, taken as factor                 | 124                 |
| 6   | city_name                            | Original feature from dataset, taken as factor                 | 257                 |
| 7   | extra_people                         | Original feature from dataset                                  | 149                 |
| 8   | first_review                         | Original feature from dataset                                  | 143                 |
| 9   | guests_included                      | Original feature from dataset                                  | 93                  |
| 10  | host_is_superhost                    | Original feature from dataset, taken as factor                 | 150                 |
| 11  | host_response_rate                   | Original feature from dataset                                  | 189                 |
| 12  | host_response_time                   | Original feature from dataset, taken as factor                 | 160                 |
| 13  | host_listings_count                  | Original feature from dataset                                  | 153                 |
| 14  | host_identity_verified               | Original feature from dataset, taken as factor                 | 152                 |
| 15  | host_since                           | Original feature from dataset                                  | 169                 |
| 16  | instant_bookable                     | Original feature from dataset, taken as factor                 | 268                 |
| 17  | is_location_exact                    | Original feature from dataset, taken as factor                 | 259                 |
| 18  | is_business_travel_ready             | Original feature from dataset, taken as factor                 | 277                 |
| 19  | price                                | Original feature from dataset, taken as log                    | 162, 234            |
| 20  | room_type                            | Original feature from dataset, taken as factor                 | 253                 |
| 21  | maximum_nights                       | Original feature from dataset, taken as log                    | 231                 |
| 22  | minimum_nights                       | Original feature from dataset, taken as log                    | 232                 |
| 23  | require_guest_phone_verification     | Original feature from dataset, taken as factor                 | 269                 |
| 24  | monthly_price                        | Original feature from dataset, taken as log                    | 164, 240            |
| 25  | no_of_amenities                      | Numerical variable created from “amenities” feature            | 127                 |
| 26  | has_notes                            | Factor variable created from “notes” feature                   | 158                 |
| 27  | has_security_deposit                 | Factor variable created from “security_deposit” feature        | 175                 |
| 28  | has_square_feet                      | Factor variable created from “square_feet” feature             | 185                 |
| 29  | is_extra_people                      | Factor variable created from “extra_people” feature            | 196                 |
| 30  | is_availability_30                   | Factor variable created from “availability_30” feature         | 200                 |
| 31  | is_availability_365                  | Factor variable created from “availability_365” feature        | 205                 |
| 32  | pratio                               | Ratio between “price” and “accommodates”                       | 217                 |
| 33  | wratio                               | Ratio between “weekly_price” and “accommodates”                | 218                 |
| 34  | mratio                               | Ratio between “monthly_price” and “accommodates”               | 219                 |
| 35  | sratio                               | Ratio between “security_deposit” and “accommodates”            | 220                 |
| 36  | aratio                               | Ratio between “square_feet” and “accommodates”                 | 221                 |
| 37  | apratio                              | Ratio between “price” and “square_feet”                        | 222                 |
| 38  | bedroomratio                         | Ratio between “bedrooms” and “accommodates”                    | 226                 |
| 39  | property_category                    | Factor variable created from “property_type” feature           | 244                 |


## Evaluation and Modeling
### Winning Model: XGBoost

```R
bst <- xgboost(data = as.matrix(data_train_x_num), 
               label = as.matrix(data_train_y_num), 
               max.depth = 2, eta = 0.2, nrounds = 700,  
               objective = "binary:logistic")
```
- **Accuracy**: ~77% on the validation dataset.
- **True Positive Rate (TPR) on the validation dataset: ~46% 

### Model Features
The XGBoost model used 39 features, including availability, bathrooms, beds, cancellation policy, city name, price, room type, and several engineered features.

- The best performing features:

<img width="425" alt="image" src="https://github.com/shahnishank/AirBnB_Ratings-Predictive_Analytics/assets/108402877/2e0f3525-3145-4bf4-b3f2-776a3cf76982">

### Hyperparameters
- **max.depth**: 2
- **eta**: 0.2
- **nrounds**: 700
- **objective**: "binary:logistic"

### Other Models
1. **Random Forest**

```R
# Random forest model
rf.mod <- ranger(x = data_train_x, y = data_train_y,
                 mtry=39, num.trees=700,
                 importance="impurity",
                 probability = TRUE)

rf_preds <- predict(rf.mod, data=data_valid_x1)$predictions[,2]
```
   - **AUC**: ~80
   - **Best Features**: first_review, pratio, instant_bookable
   - **Hyperparameters**: mtry = 39, num.trees = 700, importance = "impurity", probability = TRUE

### ROC Curve for the 2 models:

<img width="423" alt="image" src="https://github.com/shahnishank/AirBnB_Ratings-Predictive_Analytics/assets/108402877/d153d5da-3262-4ea0-87ce-f116409a9f98">

## Reflection and Takeaways
### Successes
- **Data Processing**: Efficient understanding and preprocessing of data.
- **Business Context Understanding**: Quick identification of use cases and effective feature selection.

### Challenges
- **Data Cleaning**: Handling extensive features and irregularities.
- **Feature Engineering**: Time-consuming but essential for effective modeling.

### Future Work
- **Improved Model Interpretability**: Enhance the interpretability of the model.
- **Integration of External Data**: Incorporate additional data sources for better predictions.

## Conclusion
This classification model is a valuable resource for various stakeholders in the Airbnb ecosystem. It enables them to make data-driven decisions, optimize operations, and improve market performance. The model's insights can increase client satisfaction, earnings, and business growth.

---

This README provides an overview of the project, highlighting key aspects such as business understanding, data preparation, modeling, and reflections on the project's outcomes and future directions. For more details, refer to the project documentation and source code.
