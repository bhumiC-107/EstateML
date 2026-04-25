**EstateML** 

A machine learning web application that predicts residential property prices in Bengaluru, India. Built with a tuned Random Forest Regressor and deployed via Streamlit, the application accepts property details as input and returns an estimated market price in Lakhs, along with a confidence range derived from the model's RMSE.

**Demo  Link : https://estateml-dre4q7qeekrjfk95v2sjzk.streamlit.app/**
**Overview**
This project addresses the challenge of estimating residential property prices in Bengaluru — one of India's fastest-growing real estate markets. The model was trained on the Bengaluru House Price dataset, which contains over 13,000 property listings with features such as area, number of bedrooms, bathrooms, and balconies.
The end-to-end pipeline covers data cleaning, feature engineering, model selection, hyperparameter tuning via GridSearchCV, and deployment as an interactive web application backed by a MySQL database for prediction logging.

**Tech Stack**
**Layer**                **Technology**
Language            Python 3.10+
Web Framework       Streamlit
Machine Learning    scikit-learn
Data Processing     pandas, NumPy
Model Persistence   joblib
Database            MySQL via SQLAlchemy
Version             ControlGit / GitHub


**Features**

1.Property price prediction based on area, BHK, bathrooms, and balconies
2.Price confidence range displayed as Estimated Price ± RMSE
3.Price per square foot computed and displayed in the result card
4.Step-by-step progress indicator reflecting the user's position in the prediction flow
5.Feature importance chart showing which property attributes drive predictions most
6.Prediction history table and trend chart backed by MySQL
7.Styled dark sidebar with gold accent theme
