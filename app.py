from flask import Flask,render_template, request
import numpy as np            
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
 import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro, kstest, normaltest, boxcox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore


import warnings
warnings.filterwarnings('ignore')


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro, kstest, normaltest, boxcox
import statsmodels.api as sm   

model = pickle.load(open('lasso_reg_model.pkl','rb'))
lasso_reg_model = Lasso(alpha = 4.99)
lasso_reg_model.fit(x_train, y_train)

app = Flask(__name__)

@app.route("/")
def my():
    return render_template("home.html" )

@app.route("/predict",methods = ["POST", "GET"])
def home():
    a = eval(request.form['area'])
    b = eval(request.form['bedrooms'])
    c = eval(request.form['bathrooms'])
    d = eval(request.form['stories'])
    e = eval(request.form['mainroad'])
    f = eval(request.form['guestroom'])
    g = eval(request.form['basement'])
    h = eval(request.form['hotwaterheating'])
    i = eval(request.form['airconditioning'])
    j = eval(request.form['parking'])
    k = eval(request.form['prefarea'])
    l = eval(request.form['furnishingstatus'])



    arr = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
    predicted_price = model.predict(arr)
    
    predicted_price = np.around(lasso_reg_model.predict([test_array]), 3)[0]
    print("predicted  price is :", predicted_price)
    return render_template('after.html', prediction_text = predicted_price)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)   
   

# @app.route('/predict1')
# def predict1():
#     a = eval(request.args.get('a'))
#     b = eval(request.args.get('b'))
#     c = eval(request.args.get('c'))
#     d = eval(request.args.get('d'))

#     arr = np.array([[a, b, c, d, e, f, g, h, i, j]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

    