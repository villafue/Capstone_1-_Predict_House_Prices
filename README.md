![cover_photo](https://github.com/villafue/Capstone_1-_Predict_House_Prices/blob/master/Pictures/Music%20Icon%20Etsy%20Banner.png)
# Predicting the Price of Homes

*[In the 2018 KPMG Global Proptech Survey](https://assets.kpmg/content/dam/kpmg/uk/pdf/2018/10/kpmg-global-proptech-survey-2018.pdf), almost 50% of the respondends believed that AI, Big Data, and Big Data Analysis, "Will have the biggest impact on the real estate industry in the long term." Furthermore, in the [Deloitte 2021 CRE Outlook](https://www2.deloitte.com/us/en/insights/industry/financial-services/commercial-real-estate-outlook.html/#endnote-22), over half (56%) of the 200 commercial real estate leaders, "Believed that digital transformation of the business could become a business imperative." In other words, there is a trend in real-estate towards harnessing and utilizing data science.* 

*In this project, I will participate in a "contest" hosted by the ficticious company AREC to predict the price of homes. Furthermore, I will present recommendations to them based on my findings and next steps for application. The link to the full notebook is below or feel free to read through the different sections.*

[Full Notebook](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb)

## 1. Data

The Ames Housing dataset was compiled by Dean De Cock and it contains 80 features with almost 3,000 rows of data. It is also considered the contemporary to the often cited [Boston Housing dataset](https://www.kaggle.com/c/boston-housing):

> * [Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project by Dean De Cock](http://jse.amstat.org/v19n3/decock.pdf)

> * [Kaggle Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

> * [Data Import Section](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#import_packages)

## 2. Method

This is a supervised machine learning problem and predictions will be made using regression analysis. 

## 3. Data Preparation 

[Data Preparation Section](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#data_preparation)

In a regression problem, the task is to predict the dependent variable given a set of independent features. The goal is measure how closely the predictions match the actual values. I had to clean and  I also had to clean & normalize all the reference information (location, difficulty grade, etc.) to the route so that my user could get a useful and informative recommendation.

* **Problem 1:** This dataset is all user-entered information. There are a couple drop down options, but for the most part the user is able to completely make-up, or list something incorrectly. **Solution:** after normalizing & cleaning all the columns, I created a three-tier groupby system that I could then take the mode of each entry and fill in the column with that mode. For example: a route listed 12 times had the country Greece associated with it 11 times, but one person incorrectly listed it located in the USA. By grouping together three other indicator columns and then computing the mode of the country, I was able to catch and change some of the user-entered errors and increase the accuracy of my dataset.

* **Problem 2:** Being this is an international rock climbing website, the names of the rock climbing routes were differing based on if the user enters accent marks or not. **Solution:** normalize all names to the ascii standards. 

* **Problem 3:** Spelling issues with the route name. For example: if there was a route named "red rocks canyon" it could be spelled "red rock", "red rocks", "red canyon" etc. **Solution:** at first, I was hopeful and tried two different phonetic spelling algorithms (soundex & double metahpone). However, both of these proved to be too aggressive in their grouping and sometimes would group together up to 20 different individual routes as the same item! My final solution was to create an accurate filter for route names. The logic being that if up to x number of users all entered that *exact same* route name, the chances were good that it was an actual route spelled correctly. I played around with 4 different filters and kept these until I could test their prediction accuracy in the ML portion. I found the greatest prediction accuracy came from the dataset that filtered out any routes listed less than 6 times.

## 4. EDA

[EDA Report](https://colab.research.google.com/drive/14AKVsyXy7yJSxBjmEBFyz7kEX7e9ioM_)

* The star-rating distributions all checked out to be normal. It is very common with explicit ratings to see a diminished number of low ratings.

![](./6_README_files/star_counts.png)

## 5. Algorithms & Machine Learning

[ML Notebook](https://colab.research.google.com/drive/1kAlvwwJnGcdCAJD8oFokT3gtJF2UnyZP)

I chose to work with the Python [surprise library scikit](http://surpriselib.com/) for training my recommendation system. I tested all four different filtered datasets on the 11 different algorithms provided, and every time the Single Value Decomposition++ (SVD++) algorithm performed the best. It should be noted that this algorithm, although the most accurate is also the most computationally expensive, and that should be taken into account if this were to go into production.

![](./6_README_files/algo.png)

>***NOTE:** I choose RMSE as the accuracy metric over mean absolute error(MAE) because the errors are squared before they are averaged which gives the RMSE a higher weight to large errors. Thus, the RMSE is useful when large errors are undesirable. The smaller the RMSE, the more accurate the prediction because the RMSE takes the square root of the residual errors of the line of best fit.*

**WINNER: SVD++ Algorithm**

This algorithm is an improved version of the SVD algorithm that Simon Funk popularized in the million dollar Netflix competition that also takes into account implicit ratings (*yj*). Using stochastic gradient descent (SGD), parameters are learned using the regularized squared error objective.

![](./6_README_files/forumla.png)

## 6. Which Dataset to choose?

[More details about this process...](https://colab.research.google.com/drive/1kAlvwwJnGcdCAJD8oFokT3gtJF2UnyZP)

After choosing the SVD++ algorithm, I tested the accuracy of all four different filtered datasets. The dataset which filtered out any route names occurring less than 6 times performed the most accurate predictions. Thus, it was chosen to be the dataset I trained on.

>* All of the dataframes displayed discrepancies with the 1 star ratings(This is to be expected due to the inherent skewed positive ratings). Also, the one star ratings are not imperative to this project's goal. It is more important that the 1 star ratings are different enough to be filtered out of the top ten routes recommended to users. 
>* Notice the 3-star rating has a fat bulge at the top of the "violin" which indicates its predicting 3-star ratings for some of the true 3-star routes. This was not as prominent in the other dataframes
>* The 1-star rating also has a fatter tail than the other datasets displayed


![](./6_README_files/accuracy.png)

## 7. Coldstart Threshold
[More details about this process...](https://colab.research.google.com/drive/1kAlvwwJnGcdCAJD8oFokT3gtJF2UnyZP)

**Coldstart Threshold**: There is a problem when only using collaborative based filtering: *what to recommend to new users with very little or no prior data?* Remember, we already set our cold start threshold for the routes by choosing the dataset that filtered out any route occurring less than 6 times. Now, let investigate where to put the threshold for users.

![](./6_README_files/20user_thresh.png)

*It is my hypothesis that the initial filtering of the routes is what affected the RMSE of the users* 

>* Increasing the user threshold to 5 would increase the RMSE by .005 & would lose approximately 40% of the data.
>* Increasing the user threshold to 13 would increase the RMSE by .0075 & would lose approximately 60% of the data
>* If there were a larger increase in the RMSE (>= .01) I would trade my users' data for this improvement. However, these improvements are too minuscule to give up 40%-60% of my data to train on. Instead, I voted to keep some of these outliers to help the model train, and will focus on fine tuning my parameters using gridsearch to improve the RMSE


## 7. Predictions

[Final Predictions Notebook](https://colab.research.google.com/drive/1vLkoW_4SYessy_igmJxlVz_jEPlgJ06v)

In the final predictions notebook, the user can enter their user_id number and receive a list of top ten routes recommended to them:

![](./6_README_files/predictions.png)

## 8. Future Improvements

* In the future, I would love to spend more time creating a filtering system, wherein a climber could filter out the type, difficulty of climb, & country before receiving their top ten recommendation

* This recommendation system could also be improved by connecting to the 8a.nu website so that the user could input their actual online ID instead of just their user_id number 

* Due to RAM constraints on google colab, I had to train a 65% sample of the original 6x dataset. Without resource limitations, I would love to train on the full dataset. Preliminary tests showed that the bigger the training size, the lower the RMSE. One test showed an increase in sample size could increase the RMSE by .03 (in contrast to the .005 improvement I received when increasing the coldstart threshold)

## 9. Credits

Thanks to Nicolas Hug for his superb surprise library scikit, Colin Brochard for his stellar advice from his Mountain Project recommendation system, and DJ Sarkar for being an amazing Springboard mentor.





