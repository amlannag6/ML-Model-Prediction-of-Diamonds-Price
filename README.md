# SUPERVISED MACHINE LEARNING TECHNIQUES for Prediction-of-Diamonds-Price 
This project mainly focused on the price prediction of diamonds based on their cut, color, clarity and price. Diamond is the highest valuable and precious metal in terms of jewellery. In northern America, Diamond has a separate Executive based market. Diamonds are perhaps the most expensive among several products that consumers do not quantitatively or objectively value. With a strong emphasis on interpersonal relations, buying is far from fair. The jewellers will entice any man and woman by selling it as a must for the occasion and a status symbol and referring to this expensive and unaffordable piece as priceless. On the other hand, a gemologist determines a diamond's real value after examining its different "features'' and applying the relative valuation theory of "compare and price." As a precious metal, diamond having the most desirable value in the current world. From my initial research on the dataset - Diamond, I analysed and found appropriate levels which would help me to build an ML agent which could predict the future price based on statistics and features. From there, I researched and finalized that I would work with this dataset. I have dropped down all unnecessary data and train them to implement various linear regression models and classifiers. For this particular model, a classifier would not be the best choice to determine the accuracy. I have implemented different types of regression models and generated accuracy results for each model, giving us a reasonable rate to predict the future diamond price rate. Understanding the background strategies of diamond price factors and future diamond values helps me determine the procedures I should take to train my agent using Machine learning techniques, which help me predict diamond price as per the requirements as diamond values depend on several features and its outreach market. Throughout the background research process, I have gathered all required techniques that would help to implement the machine learning techniques. I also analyzed the  4 dataset and implemented all regression models to determine the best accuracy and best regression value to predict the diamond price.
[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/rasa.svg)](https://badge.fury.io/py/rasa)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa.svg)](https://pypi.python.org/pypi/rasa)
[![Build Status](https://github.com/RasaHQ/rasa/workflows/Continuous%20Integration/badge.svg)](https://github.com/RasaHQ/rasa/actions)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa/badge.svg?branch=main)](https://coveralls.io/github/RasaHQ/rasa?branch=main)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)
![Documentation Build](https://img.shields.io/netlify/d2e447e4-5a5e-4dc7-be5d-7c04ae7ff706?label=Documentation%20Build)
[![FOSSA Status](https://app.fossa.com/api/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git.svg?type=shield)](https://app.fossa.com/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git?ref=badge_shield)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/orgs/RasaHQ/projects/23)
# Project Details
I chose the dataset of “Diamonds” As described in the introduction, I analyzed the dataset
initially. I found that there are levels in the dataset: “ Index #, carat, cut, color, clarity, depth, table,
price, x, y, z.” I looked at the Diamonds dataset, which includes approximately 54,000 diamond
prices and other attributes like carat, cut. There are 53940 rows and ten variables in this dataset.
From this data set, I can train the agent in Machine Learning’s environment to observe the price
as per quality and then it could predict.
Here all attributes are followed by :
1. Price is in US dollars and Carat weight of the diamond.
2. Cut quality: Fair, Good, Very Good, Premium, Ideal and Color of the diamond, with D
being the best and J the worst.
3. Clarity : (In order from best to worst, FL = flawless, I3= level 3 inclusions) IF, VVS1 and
Depth %: The height of a diamond, measured from the culet to the table, divided by its
average girdle diameter.
4. table%: The width of the diamond's table is expressed as a percentage of its average
diameter and x length in mm and y width in mm.
Initially, I started sorting various rows per level and then determined that the index level
would not require the project. Because the database has an automated number-wise level index as
per data values. So, I dropped the column Unnamed=0; then it’s representing all data with
numbered values. Then, I have called functions and methods to display data values and attributes.
Then, I have described the data and then generated graphs and plots. Then implement the function 
5
data.isnull().sum() returns the number of missing values in the data set. Then after encoding the
column values, it seems that for this project, there is no need for the column -depth, table, x,y,z as
the prediction will focus on price carat, caret value, and allotment frequency would be enough to
implement machine learning. So, I have dropped column depth, table, x,y and z. After that, I
changed all data attributes to float. Then based on that, implement virtualization graphical views
of allotment of Diamond price.
This machine learning project that will help predict the diamond value would implement
regression and classification. Accuracy and R^2 regression score function that coefficient of
determination will help particular importance. Here in this project mainly used Linear Regression
Algorithm, the regression algorithm, I have implemented the determinesLinearRegression model,
DecisionTreeRegressor, Lasso, RandomForestRegressor, here RandomForestRegressor have been
performed most accurately.
Splitting arrays or matrices into random train and test subsets, I have implemented the data
arrays, shuffle values, and listed return values. It would then transform to standard scaler to x_train
and x_test and fit the transform to classifier and regression model.
To suit the results, train the model. Linear regression's fundamental aim is to minimize the
cost function. Before developing the model, it is a start to transform the categorical data to
numerical data. There are two approaches to do this. 1. Integer Encoding or Mark Encoder 2. It is
encoding in a single pass. For preparation, we need to translate the data from the Pandas data frame
into PyTorch tensors. The first step is to transform it to NumPy arrays. For training and validation,
we will need to build PyTorch datasets and data loaders. The first step is to make a TensorDataset
by translating the input and target arrays to tensors with the torch.tensor function. And now, we'll 
6
divide the dataset into two parts: training and validation. The training set will be used to train the
model, while the validation set will be used to test the model and tune the hyperparameters for
improved generalization of the trained model. In the end, the user will input their acceptations like
value for carat, cut, clarity and color; then, the machine will predict the price. 
