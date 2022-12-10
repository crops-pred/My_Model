import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    data = pd.read_csv('Crop_recommendation.csv')

    y = data['label']
    x = data.drop(['label'],axis = 1)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    prediction = model.predict((np.array([[90,
                                       42,
                                       43,
                                       20.87974371,
                                       82.00274423,
                                       6.502985292000001,
                                       202.9355362]])))
    print("The suggested Crop for Given Climatic condition is :", prediction)