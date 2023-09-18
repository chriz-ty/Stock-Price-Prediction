import numpy as np
import quandl, math, datetime
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot') #defining the style of the plots.

df = quandl.get('WIKI/GOOGL')  #fetching the data.
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#creating two new columns for calculating price voltality and price change.

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'  #sets the target variable that is to be predicted.
df.fillna(-99999, inplace=True) #It fills any missing data in the DataFrame with the value -99999.

forecast_out = int(math.ceil(0.01*len(df))) #calculating the number of days to be predicted.

df['label'] = df[forecast_col].shift(-forecast_out)  #This column represents the future prices to be predicted.

x = np.array(df.drop(['label'], axis=1))  #preparing the feature datas by dropping the label column.
x = preprocessing.scale(x)

#The next lines separates the most recent 'forecast_out' days of data as 'x_lately,' which will be used for making future predictions.
# The rest of the data is stored in 'x' for training and testing.
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]

df.dropna(inplace=True) #removes the rows with missing values.
y = np.array(df['label']) #creating the target variable.

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

#TRAINING SECTION (is commented out since the model is already trained).
'''
clf = LinearRegression(n_jobs=10) #initializes a Linear Regression model and uses 10 CPU cores in parallel when fitting the model.
clf.fit(x_train, y_train)  #fits the Linear Regression model to the training data.

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)   #saving to a pickle file
'''

pickle_in = open('linearregression.pickle','rb')  #It loads the pre-trained linear regression model from the pickle file.
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test) #It calculates the accuracy of the model.
forecast_set = clf.predict(x_lately) #uses the loaded model to make predictions for the 'x_lately' data.
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan  #uses the loaded model to make predictions for the 'x_lately' data.

#setting up variables to keep track of time for adding future dates and predictions to the DataFrame.
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


#plotting the historical stock prices.
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
