![cover_photo](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/Music%20Icon%20Etsy%20Banner%20(1).png)
# Predicting the Price of Homes

*[In the 2018 KPMG Global Proptech Survey](https://assets.kpmg/content/dam/kpmg/uk/pdf/2018/10/kpmg-global-proptech-survey-2018.pdf), almost 50% of the respondents believed that AI, Big Data, and Big Data Analysis, "Will have the biggest impact on the real estate industry in the long term." Furthermore, in the [Deloitte 2021 CRE Outlook](https://www2.deloitte.com/us/en/insights/industry/financial-services/commercial-real-estate-outlook.html/#endnote-22), over half (56%) of the 200 commercial real estate leaders, "Believed that digital transformation of the business could become a business imperative." In other words, there is a trend in real-estate towards harnessing and utilizing data science.* 

*In this project, I will participate in a "contest" hosted by the ficticious company AREC to predict the price of homes. Furthermore, I will present recommendations to them based on my findings and next steps for application. The link to the full notebook is below, or feel free to read through the different sections.*

[[Full Notebook]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb)

## 1. Data

The Ames Housing dataset was compiled by Dean De Cock and it contains 80 features with almost 3,000 rows of data. It is also considered the contemporary to the often cited [Boston Housing dataset](https://www.kaggle.com/c/boston-housing):

> * [Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project by Dean De Cock](http://jse.amstat.org/v19n3/decock.pdf)

> * [Kaggle Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

> * [Data Import Section](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#import_packages)

## 2. Method

This is a supervised machine learning problem and predictions will be made using regression analysis. 

## 3. Data Preparation 

[[Data Preparation Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#data_preparation)

In a regression problem, the task is to predict the dependent variable given a set of independent features. The goal is measure how closely the predictions match the actual values. To prepare my data, I had to put both train and test sets together, remove outliers, and impute the missing values. 

## 4. EDA

[[EDA Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#exploratory_data_analysis)

* I went through each of the 80 features and analyzed their pattern against the price of the homes.

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/EDA_BsmtQual.png)

## 5. Pre-Processing

[[Pre-Processing]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#target_variable)

* I transformed the dependent variable into something that resembles a more normal distribution, as well as corrected skewness for the indendent features. By transforming the features, it helped in prediction especially for the linear-based models. 

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/SalePrice%20Transformed.png)

## 6. Modeling

[[Modeling Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#modeling)

I chose to use Python's [scikit-learn library](https://scikit-learn.org/stable/) for my regression problem. I prediced the price of homes using a mixture of linear-based and tree-based algorithms against a baseline model. Once the models were optimized to my data, I then used them as input for my meta-stacking algorithm.

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/Final%20Table.png)

>***NOTE:** I choose RMSE as the accuracy metric over mean absolute error(MAE) because the errors are squared before they are averaged which gives the RMSE a higher weight to large errors. Thus, the RMSE is useful when large errors are undesirable. The smaller the RMSE, the more accurate the prediction because the RMSE takes the square root of the residual errors of the line of best fit. Furthermore, it is also the chosen metric for how my predictions were to be scored.*

**WINNER: [Stacking Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)**

This algorithm inputs models as base predictors, and by default, uses a version of the [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV) algorithm as the final predictor. The optimized version used both my models and the original training set as input for prediction.  

## 7. Predictions

[[Project Report]](https://github.com/villafue/Capstone_1-_Predict_House_Prices/blob/master/Final/Capstone%201%20Final%20Report%203Feb21.pdf)

Upon submission, I had a final RMSE score of 0.12676 which means that, on average, my predicted house prices were $0.13 off for every $1.00. I also included three business recommendations, for the AREC CEO, on how to prepare her company to use this data.

>***NOTE:** The recommendations are under the "Recommendations" section in the "Report" link above.*

## 8. Future Improvements

[[Project Report]](https://github.com/villafue/Capstone_1-_Predict_House_Prices/blob/master/Final/Capstone%201%20Final%20Report%203Feb21.pdf)

* Compare Time Series Analysis with inflation data. There seemed to be a positive correlation of the price of homes against the year it was built. However, the data was not adjusted for inflation so that is something for further exploration.

* Experiment with Bayesian Hyperparameter Tuning. GridSearch is computationally expensive and RandomSearch is a game of chance. Bayesian Optimization is essentially a “smart” RandomSearch that can identify areas that can improve the predictive accuracy.   

* Auto Modeling and Tuning. In [Section 6.4](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#TPOT), I showed my attempt at using the [TPOT](http://epistasislab.github.io/tpot/) automated machine learning tool. The meta-model it chose was very convoluted and had its own API. I would like to explore this more and see how well it ultimately scores.

>***NOTE:** For the full list of future improvements, please go to the "Areas for Further Exploration" section in the "Report" link above.*

## 9. Credits

This project was a huge labor of love and could not have been accomplished without help and guidance. First, Thanks to Pedro, Serigne, Bsivavenu, Jesuscristo, Pavan, and Arun for their amazing notebooks! I drew a lot of inferences, strategies, and insights from them. Links to their work can be seen under the [6.3 - References](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#references) of my notebook. Also, thank you to Kenneth Gil Pascual for his mentoring and guidance has been absolutely invaluable to the quality and completion of this project.




