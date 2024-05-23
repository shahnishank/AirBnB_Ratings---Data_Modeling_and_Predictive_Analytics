#load libraries
library(tidyverse)
library(dplyr)
library(caret)
library(openxlsx)
library(pROC)
library(ROCR)
library(ranger)
library(xgboost)

setwd("C:/Tan/Data Mining/Project")
options(scipen = 999)

#load files
train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")

#Variable check
table(train_x$accommodates)
table(train_y$perfect_rating_score, train_x$accommodates)
table(train_x$availability_30)
table(train_y$perfect_rating_score, train_x$availability_30)
table(train_x$availability_365)
table(train_y$perfect_rating_score, train_x$availability_365)
table(train_x$availability_60)
table(train_y$perfect_rating_score, train_x$availability_60)
table(train_x$availability_90)
table(train_y$perfect_rating_score, train_x$availability_90)
table(train_x$bathrooms)
table(train_y$perfect_rating_score, train_x$bathrooms)
table(train_x$bed_type)
table(train_y$perfect_rating_score, train_x$bed_type)
table(train_x$bedrooms)
table(train_y$perfect_rating_score, train_x$bedrooms)
table(train_x$beds)
table(train_y$perfect_rating_score, train_x$beds)
table(train_x$cancellation_policy)
table(train_y$perfect_rating_score, train_x$cancellation_policy)
table(train_x$city_name)
table(train_y$perfect_rating_score, train_x$city_name)
table(train_x$cleaning_fee)
table(train_y$perfect_rating_score, train_x$cleaning_fee)
table(train_x$guests_included)
table(train_y$perfect_rating_score, train_x$guests_included)
summary(train_x$host_is_superhost)
table(train_y$perfect_rating_score, train_x$host_is_superhost)
summary(train_x$host_listings_count)
table(train_y$perfect_rating_score, train_x$host_listings_count)
table(train_x$instant_bookable)
table(train_y$perfect_rating_score, train_x$instant_bookable)
summary(train_x$is_business_travel_ready)
table(train_y$perfect_rating_score, train_x$is_business_travel_ready)
summary(train_x$maximum_nights)
table(train_y$perfect_rating_score, train_x$maximum_nights)
summary(train_x$minimum_nights)
table(train_y$perfect_rating_score, train_x$minimum_nights)
summary(train_x$require_guest_phone_verification)
table(train_y$perfect_rating_score, train_x$require_guest_phone_verification)
summary(train_x$require_guest_profile_picture)
table(train_y$perfect_rating_score, train_x$require_guest_profile_picture)
summary(train_x$security_deposit)
table(train_y$perfect_rating_score, train_x$security_deposit)

#Combine train and test for cleaning
train_y <- train_y %>% select(perfect_rating_score)%>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score))
train_x <- train_x %>% mutate(dataset = "train")
test_x <- test_x %>% mutate(dataset = "test")
combined <- rbind(train_x, test_x)

# EXAMPLE PREDICTIONS FOR CONTEST 1

# Select required variables and perform cleaning
train_perfect <- combined %>%
  select(c(access,
           accommodates, 
           amenities, 
           availability_30,
           availability_365,
           availability_60,
           availability_90,
           bathrooms, 
           bed_type, 
           bedrooms, 
           beds, 
           cancellation_policy, 
           city_name, 
           cleaning_fee, 
           description,
           extra_people, 
           first_review,
           guests_included, 
           host_about,
           host_is_superhost, 
           host_acceptance_rate,
           host_response_rate,
           host_response_time,
           host_listings_count,
           host_identity_verified,
           host_since,
           host_verifications,
           instant_bookable,
           interaction,
           is_location_exact, 
           is_business_travel_ready,
           price, 
           property_type,
           room_type, 
           notes,
           maximum_nights,
           minimum_nights,
           require_guest_phone_verification,
           require_guest_profile_picture,
           requires_license,
           security_deposit,
           square_feet,
           host_listings_count,
           weekly_price,
           monthly_price,
           dataset)) %>% 
  
  # Group strict, super_strict_30 and super_strict_60 into strict
  mutate(cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30", "super_strict_60"), "strict", cancellation_policy)) %>%
  
  # New variable: no_of_amenities
  mutate(no_of_amenities = ifelse(amenities == "{}", 0, ifelse(str_detect(amenities, ","), str_count(amenities, ",") + 1, 1))) %>%
  
  # New variable: no_host_verifications
  mutate(no_host_verifications = ifelse(host_verifications == "[]", 0, ifelse(str_detect(host_verifications, ","), str_count(host_verifications, ",") + 1, 1))) %>%
  
  # Convert to numeric
  mutate(cleaning_fee = as.numeric(cleaning_fee), 
         price = as.numeric(price),
         weekly_price = as.numeric(weekly_price),
         monthly_price = as.numeric(monthly_price),
         extra_people = as.numeric(extra_people),
         security_deposit = as.numeric(security_deposit ),
         square_feet = as.numeric(square_feet),
         host_response_rate = as.numeric(host_response_rate),
         host_acceptance_rate = as.numeric(host_acceptance_rate),
         host_since = as.numeric(host_since),
         first_review = as.numeric(first_review)
  ) %>%
  
  # Replace NAs
  mutate(
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    extra_people = ifelse(is.na(extra_people), 0, extra_people),
    host_is_superhost = ifelse(is.na(host_is_superhost), "FALSE", host_is_superhost),
    is_business_travel_ready = ifelse(is.na(is_business_travel_ready), "FALSE",is_business_travel_ready),
    host_identity_verified = ifelse(is.na(host_identity_verified), "FALSE", host_identity_verified),
    host_listings_count = ifelse(is.na(host_listings_count), 1, host_listings_count), 
    no_of_amenities = ifelse(is.na(no_of_amenities), 0, no_of_amenities),
    no_host_verifications = ifelse(is.na(no_host_verifications), 0, no_host_verifications),
    has_host_about = ifelse(is.na(host_about), "FALSE", "TRUE"),
    has_access = ifelse(is.na(access), "FALSE", "TRUE"),
    has_notes = as.factor(ifelse(is.na(notes), "FALSE", "TRUE")),
    has_desc = ifelse(is.na(description), "FALSE", "TRUE"),
    host_response_time = ifelse(is.na(host_response_time), "OTHER", host_response_time)) %>%
  
  mutate(price = ifelse(is.na(price),median(price, na.rm = TRUE), price),
         weekly_price = ifelse(is.na(weekly_price),median(weekly_price, na.rm = TRUE), weekly_price),
         monthly_price = ifelse(is.na(monthly_price),median(monthly_price, na.rm = TRUE), monthly_price),
         bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms),
         beds = ifelse(is.na(beds), median(beds, na.rm = TRUE), beds),
         bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
         accommodates = ifelse(is.na(accommodates), median(accommodates, na.rm = TRUE), accommodates),
         host_since = ifelse(is.na(host_since), median(host_since, na.rm = TRUE), host_since),
         first_review = ifelse(is.na(first_review), median(first_review, na.rm = TRUE), first_review),
         availability_30 = ifelse(is.na(availability_30), median(availability_30, na.rm = TRUE), availability_30),
         availability_60 = ifelse(is.na(availability_60), median(availability_60, na.rm = TRUE), availability_60),
         availability_90 = ifelse(is.na(availability_90), median(availability_90, na.rm = TRUE), availability_90),
         availability_365 = ifelse(is.na(availability_365), median(availability_365, na.rm = TRUE), availability_365),
         has_security_deposit = as.factor(cut(security_deposit, 
                                              breaks=c(-Inf, 0,100, 200, 400, Inf),
                                              labels=c("No-Deposit","Low", "Medium-Low", "Medium", "High"),
                                              include.lowest=TRUE)),
         has_security_deposit = as.factor(ifelse(is.na(has_security_deposit), "Not Available", has_security_deposit)),
         security_deposit = ifelse(is.na(security_deposit),median(security_deposit, na.rm = TRUE), security_deposit),
         has_square_feet = as.factor(cut(square_feet, 
                                        breaks=c(-Inf,350, 750, 950, Inf),
                                       labels=c("Small", "Medium-Small", "Medium", "Large"),
                                       include.lowest=TRUE)),
         has_square_feet = as.factor(ifelse(is.na(has_square_feet), "False", "True")), 
         has_int = as.factor(ifelse(is.na(interaction), "False", "True")), 
         square_feet = ifelse(is.na(square_feet),median(square_feet, na.rm = TRUE),square_feet),
         host_acceptance_rate = ifelse(is.na(host_acceptance_rate),median(host_acceptance_rate, na.rm = TRUE), host_acceptance_rate),
         host_response_rate= ifelse(is.na(host_response_rate),median(host_response_rate, na.rm = TRUE), host_response_rate),
         guests_included_bin = as.factor(cut(guests_included, 
                                              breaks=c(-Inf, 0,5, 10, 15, Inf),
                                              labels=c("No Guests","Low", "Medium-Low", "Medium", "High"),
                                              include.lowest=TRUE))) %>%
  
  # New variable: is_extra_people, is_guests_included
  mutate(is_extra_people = ifelse(extra_people == 0, "NO", "YES")) %>%
  mutate(is_guests_included = ifelse(guests_included == 0, "NO", "YES")) %>%
  
  # New variable: is_availability_30
  mutate(is_availability_30 = ifelse(availability_30 == 0, "NO", "YES")) %>%
  # New variable: is_availability_60
  mutate(is_availability_60 = ifelse(availability_60 == 0, "NO", "YES")) %>%
  # New variable: is_availability_90
 mutate(is_availability_90 = ifelse(availability_90 == 0, "NO", "YES")) %>%
  # New variable: is_availability_365
  mutate(is_availability_365 = ifelse(availability_365 == 0, "NO", "YES")) %>%
  
  # New variable: has_cleaning_fee
  mutate(has_cleaning_fee = ifelse(cleaning_fee > 0, "YES", "NO")) %>%
  
  # New variable: bed_category
  mutate(bed_category = ifelse(bed_type == "Real Bed", "bed", "other")) %>%
  
  # New variables: Ratios
  mutate(bbratio = bathrooms * bedrooms,
         egratio = guests_included * extra_people,
         pratio = ifelse(accommodates != 0, price/accommodates, 0),
         wratio = ifelse(accommodates != 0, weekly_price/accommodates, 0),
         mratio = ifelse(accommodates != 0, monthly_price/accommodates, 0),
         sratio = ifelse(accommodates != 0, security_deposit/accommodates, 0),
         aratio = ifelse(accommodates != 0, square_feet/accommodates, 0),
         apratio = ifelse(square_feet != 0, price/square_feet, 0),
         awratio = ifelse(square_feet != 0, weekly_price/square_feet, 0),
         amratio = ifelse(square_feet != 0, monthly_price/square_feet, 0),
         bedratio = ifelse(accommodates != 0, beds/accommodates, 0),
         bedroomratio = ifelse(accommodates != 0, bedrooms/accommodates, 0),
         bathroomratio = ifelse(accommodates != 0, bathrooms/accommodates, 0),
         avnights = maximum_nights - minimum_nights,
         avnights = log(avnights +1),
         abratio = accommodates*beds,
         maximum_nights = log(maximum_nights + 1),
         minimum_nights = log(minimum_nights + 1),
         price_dev = price - mean(price),
         price = log(price + 1),
         weekly_price = log(weekly_price + 1),
         aratio = log(aratio +1),
         apratio = log(apratio +1),
         awratio = log(awratio +1),
         amratio = log(amratio +1),
         monthly_price = log(monthly_price + 1),
         host_listings_count = log(host_listings_count +1)) %>%
  
  # New variable: property_category
  mutate(property_category = case_when(
    property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
    property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
    property_type %in% c("Townhouse", "Condominium") ~ "condo",
    property_type %in% c("Bungalow", "House") ~ "house",
    TRUE ~ "other")) %>%
  
  # Convert to factor
  mutate(cancellation_policy = as.factor(cancellation_policy),
         room_type = as.factor(room_type),
         property_category = as.factor(property_category),
         host_is_superhost = as.factor(host_is_superhost),
         host_identity_verified = as.factor(host_identity_verified),
         city_name = as.factor(city_name),
         has_cleaning_fee = as.factor(has_cleaning_fee),
         is_location_exact = as.factor(is_location_exact),
         property_type = as.factor(property_type),
         bed_category = as.factor(bed_category),
         is_extra_people = as.factor(is_extra_people),
         is_guests_included = as.factor(is_guests_included),
         is_availability_30 = as.factor(is_availability_30),
         is_availability_60 = as.factor(is_availability_60),
         is_availability_90 = as.factor(is_availability_90),
         is_availability_365 = as.factor(is_availability_365),
         instant_bookable = as.factor(instant_bookable),
         require_guest_phone_verification = as.factor(require_guest_phone_verification),
         require_guest_profile_picture = as.factor(require_guest_profile_picture),
         requires_license = as.factor(requires_license),
         has_security_deposit = as.factor(has_security_deposit),
         has_host_about = as.factor(has_host_about),
         has_access = as.factor(has_access),
         has_desc = as.factor(has_desc),
         host_response_time = as.factor( host_response_time),
         is_business_travel_ready = as.factor(is_business_travel_ready)
  ) %>%

  # Select required variables for model
  select(c(availability_30,                 
           availability_365,
           bathrooms,                       
           #bedrooms,                        
           beds,          
           cancellation_policy,
           city_name,    
           #egratio,
           extra_people,                   
           first_review,                 
           guests_included,   
           #guests_included_bin,
           #has_host_about,
           host_is_superhost,            
           #host_acceptance_rate,            
           host_response_rate,              
           host_response_time,             
           host_listings_count,          
           host_identity_verified,
           host_since,              
           #no_host_verifications,
           instant_bookable,               
           is_location_exact,        
           is_business_travel_ready,
           price,                       
           room_type,
           maximum_nights,                  
           minimum_nights,                  
           require_guest_phone_verification,
           monthly_price,
           dataset,                 
           no_of_amenities,
           #has_access,                      
           has_notes,   
           #weekly_price,
           has_security_deposit,
           has_square_feet,    
           #has_int,
           is_extra_people,                 
           is_availability_30,             
           is_availability_365,
           #bed_category,
           #bbratio,                         
           pratio,    
           #price_dev,
           wratio,                          
           mratio,                          
           sratio,                          
           aratio,                         
           apratio,                         
           #amratio,
           bedroomratio,                
           property_category))


summary(train_perfect)

# Split train and test
train_x <- train_perfect[train_perfect$dataset == "train", ]
test_x <- train_perfect[train_perfect$dataset == "test", ]
train_x <- subset(train_x ,  select = -which(names(train_x) == "dataset"))
test_x <- subset(test_x ,  select = -which(names(test_x) == "dataset"))


# Split train data into train, valid1 and valid2 to find accuracy
set.seed(33)
train_insts = sample(nrow(train_x), .6*nrow(train_x))

data_train_x <- train_x[train_insts,]
data_valid_x <- train_x[-train_insts,]
data_train_y <- train_y[train_insts,]$perfect_rating_score
data_valid_y <- train_y[-train_insts,]

valid_insts = sample(nrow(data_valid_x), .5*nrow(data_valid_x))
data_valid_x1 <- data_valid_x[valid_insts,]
data_valid_x2 <- data_valid_x[-valid_insts,]
data_valid_y1 <- data_valid_y[valid_insts,]$perfect_rating_score
data_valid_y2 <- data_valid_y[-valid_insts,]$perfect_rating_score


# Check p values using logistic model
data_train <-  cbind(data_train_x, data_train_y) 
data_valid <-  cbind(data_valid_x2, data_valid_y2)
logistic_model <- glm(formula = data_train_y ~ .,
                      data = data_train, 
                      family = "binomial")
summary(logistic_model)

# Random forest model
rf.mod <- ranger(x = data_train_x, y = data_train_y,
                 mtry=39, num.trees=700,
                 importance="impurity",
                 probability = TRUE)

rf_preds <- predict(rf.mod, data=data_valid_x1)$predictions[,2]


# Calculate AUC
preds_rf <- prediction(rf_preds, data_valid_y1)
perf_rf <- performance(preds_rf, measure = "auc")
auc_rf <- perf_rf@y.values[[1]]
auc_rf
rf_perf <- performance(preds_rf, 
                       measure = "tpr", 
                       x.measure = "fpr")
rf_perf_tpr = unlist(rf_perf@y.values)
rf_perf_fpr = unlist(rf_perf@x.values)


# XGBOOST model

# convert data to numeric 
data_train_y_num <- as.numeric(data_train_y) - 1 
data_valid_y1_num <- as.numeric(data_valid_y1) - 1 
data_valid_y2_num <- as.numeric(data_valid_y2) - 1 
dv <- dummyVars(~ ., data = data_train_x, fullRank = TRUE)
data_train_x_num <- data.frame(predict(dv, newdata = data_train_x ))
dv2 <- dummyVars(~ ., data = data_valid_x1, fullRank = TRUE)
data_valid_x1_num <- data.frame(predict(dv2, newdata = data_valid_x1 ))
dv4 <- dummyVars(~ ., data = data_valid_x2, fullRank = TRUE)
data_valid_x2_num <- data.frame(predict(dv4, newdata = data_valid_x2 ))

bst <- xgboost(data = as.matrix(data_train_x_num), 
               label = as.matrix(data_train_y_num), 
               max.depth = 2, eta = 0.2, nrounds = 700,  
               objective = "binary:logistic")

# Valid 1
preds_bst <- predict(bst, as.matrix(data_valid_x1_num))
preds_roc <- prediction(preds_bst, data_valid_y1_num)
classifications_bst <- ifelse(preds_bst > 0.5075, 1, 0)

# Calculate accuracy
acc_bst <- mean(ifelse(classifications_bst == data_valid_y1_num, 1, 0))
acc_bst

bst_perf <- performance(preds_roc, 
                        measure = "tpr", 
                        x.measure = "fpr")
bst_perf_tpr = unlist(bst_perf@y.values)
bst_perf_fpr = unlist(bst_perf@x.values)

# TPR at 10% fpr
cutoff_index <- max(which(bst_perf_fpr < 0.098))
bst_perf@alpha.values[[1]][cutoff_index]
tpr_at_cutoff <- bst_perf_tpr[cutoff_index]
tpr_at_cutoff
fpr_at_cutoff <- bst_perf_fpr[cutoff_index]
fpr_at_cutoff

# Valid 2
preds_bst2 <- predict(bst, as.matrix(data_valid_x2_num))
preds_roc2 <- prediction(preds_bst2, data_valid_y2_num)
classifications_bst2 <- ifelse(preds_bst2 > 0.5075, 1, 0)

# Calculate accuracy
acc_bst2 <- mean(ifelse(classifications_bst2 == data_valid_y2_num, 1, 0))
acc_bst2

bst_perf2 <- performance(preds_roc2, 
                         measure = "tpr", 
                         x.measure = "fpr")
bst_perf_tpr2 = unlist(bst_perf2@y.values)
bst_perf_fpr2 = unlist(bst_perf2@x.values)

# TPR at 10% FPR
cutoff_index2 <- max(which(bst_perf_fpr2 < 0.098))
bst_perf2@alpha.values[[1]][cutoff_index2]
tpr_at_cutoff2 <- bst_perf_tpr2[cutoff_index2]
tpr_at_cutoff2
fpr_at_cutoff2 <- bst_perf_fpr2[cutoff_index2]
fpr_at_cutoff2

# Plot ROC Curve
plot(bst_perf_fpr ,type="l", bst_perf_tpr, col = "red",lwd=2, 
     main = "ROC Curve", ylim = c(0, 0.6), xlim = c(0, 0.2),
     xlab="False Positive Rate", ylab="True Positive Rate")
lines(rf_perf_fpr, rf_perf_tpr, col="blue", lwd=2)
lines(bst_perf_fpr2 , bst_perf_tpr2, col = "orange",lwd=2)
legend("bottomright", legend = c("Xgboost Valid 1", 
                                 "Xgboost Valid 2",
                                 "Random Forest"),  
                                 col = c("red", "orange","blue"), lwd = 2)

# Get feature importance scores
importance_scores <- xgb.importance(colnames(data_train_x_num), model = bst)

# Print feature importance scores
print(importance_scores)


# Test
dv3 <- dummyVars(~ ., data = test_x, fullRank = TRUE)
data_test_x_num <- data.frame(predict(dv3, newdata = test_x ))

preds_bst_test <- predict(bst, as.matrix(data_test_x_num))
classifications_bst_test <- ifelse(preds_bst_test > 0.5075, "YES", "NO")
classifications_bst_test <- ifelse(is.na(classifications_bst_test), "NO", classifications_bst_test)
write.table(classifications_bst_test, "perfect_rating_score_group8.csv", row.names = FALSE)
