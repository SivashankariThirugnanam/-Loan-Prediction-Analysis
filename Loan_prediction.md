# Loan-Prediction Analysis
 
Problem Statement:
Loans are the core business of banks. The main profit comes directly from the loan’s interest. The loan companies grant a loan after an intensive process of verification and validation. However, they still do not have assurance if the applicant is able to repay the loan with no difficulties. The two most critical questions in the lending industry are:

How risky is the borrower?
Given the borrower’s risk, should we lend him/her?
In the modern era, the data science teams in the banks build predictive models using machine learning to predict how likely a client is going to default the loan when they only have a handful of information. Loan Prediction is a very common real-life problem that each retail bank faces at least once in its lifetime. If done correctly, it can save a lot of man hours at the end of a retail bank.

# Goal of this Project:

Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban, and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form.

To automate this process, they have given a problem to identify the customers’ segments, those are eligible for loan amount so that they can specifically target these customers. The goal of this project is to predict whether a loan would be approved or not.

# Overview 

The Dataset contains the following columns such as,

Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,
Property_Area,Loan_Status.

The Dataset contains complete of 614 rows and 13 columns.

Based on the dataset we perform descriptive statistics,Exploratory Data Analysis, Creating of new attributes, Correlation matrix, Label encoding, Feature scaling,and predictive modelling.


# Importing the Libraries

#for numerical operations
import numpy as np

#for dataframe operations
import pandas as pd

#for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

#for machine learning algorithms
import sklearn

#Version of Jupyter Notebook  - v6.5.3

#Interface 

Statistical Programming Python -  3.9.7


# Feature Engineering:

Based on the domain knowledge, we can come up with new features that might affect the target variable. We can come up with following new three features:

Total Income: As evident from Exploratory Data Analysis, we will combine the Applicant Income and Coapplicant Income. If the total income is high, chances of loan approval might also be high.

EMI: EMI is the monthly amount to be paid by the applicant to repay the loan. Idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate EMI by taking the ratio of loan amount with respect to loan amount term.

Balance Income: This is the income left after the EMI has been paid. Idea behind creating this variable is that if the value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval.

Let us now drop the columns which we used to create these new features. Reason for doing this is, the correlation between those old features and these new features will be very high and logistic regression assumes that the variables are not highly correlated. We also want to remove the noise from the dataset, so removing correlated features will help in reducing the noise too.


# Conclusion:

Explored the dataset to understand the data.

Perform different tests of statistical significance to uncover hidden data relationships that our predictive model could learn from and leverage when predicting unseen instances.

Employed statistical analysis using various python libraries to identify the number and names of features that could more likely help in identifying the potential customers by predicting whether a loan would be approved or not.
