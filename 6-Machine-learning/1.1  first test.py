import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC         # stand for support vector classiffier


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Le dataset importer parle de rayon lumineux d'un télescope
#Les class sont g pour gamma et f pour l'autre
#Toute la doc sur:  https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
#Le reste des colonnes sont les features que l'on mettras dans le model

"""
Different type of machine learning :

    01) Supervised Learning:
        For all input we have the output so the model can compare
        what he calculate and what is the real answer in order to learn

        Learning task:
            Classification:
                Classification                  Pizza/ Keba/ Orange
                Binary classificatiion          Spam/not Spam

            Regression:
                Predict continues values ex: crypto 

    02) Unsupervised learning 
        The goal for the computer is to regonize pattern 
        He dont provide an anwser but put all the data in some categories
        Bassed on what he calculated

    03) Renforcement Learning
        Based on reward an punition 
        We reward the computer for doing something that may be good
"""


"""
    Bunch of input that go into a model
    And the ouput is a prediction
    noted Y

    All the input are called features
    And are noted X
    
    Different types of features:
        Qualitative :   Categorical data ex:(gender) 
            It can have an order or not ex: rating can have number but not nationality

        Quantitative : Numerical valued date

"""

"""
    We divide the datasets into 3 parts

    The training datasets
    The validation dataset
    The testing dataset

"""
"""
    Type of Loss

    L1 Loss:    loss = sum(abs(y_real - y_predicted))
    L2 Loss:    loss = sum((y_real - y_predicted)**2)
    Binary Cross entropy Loss  ==> un bordel
"""

#L'on ajoute le nom des colonnes à chaque colonne
cols = ("fLength", "fWidth", "fSize", "fConc" , "fConc1" , "fAsym", "fM3Long", "fM3Trans" , "fAlpha" , "fDist" , "class")
df =pd.read_csv("magic04.data", names=cols)

#class est soit g ou f mais du coup on transformme en binaire 
df["class"] = (df["class"] == "g").astype(int)
#print(df.head())

for label in cols[:-1]:
    plt.hist(df[df['class']==1][label], color = 'blue', label = 'gamma', alpha = 0.7 , density = True)
    plt.hist(df[df['class']==0][label], color = 'red', label = 'hadron', alpha = 0.7 , density = True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show()

train , valid , test = np.split(df.sample(frac=1), [int(0.6* len(df)), int(0.8* len(df))])

def scale_dataset(dataframe , oversample = False):
    X = dataframe[dataframe.columns[:-1]].values
    Y = dataframe[dataframe.columns[-1]].values
    #X is 2D    Y is 1D

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #L'on peut ajouter des samples alléatoire en cas de mauvaise répartition des résultats voulu
    if oversample:
        ros = RandomOverSampler()

        #On veut autant de cas ou class == 0 que class == 1 et c'est ce que fait cette ligne
        X , Y = ros.fit_resample(X,Y)

    #We turn the whole data set into an array 
    #We also tranform Y into a 2D array
    data = np.hstack((X, np.reshape(Y , (-1,1))))
                     
    return data, X, Y

train , X_train , Y_train = scale_dataset(train , oversample = True)

#L'on a pas besoin d'équilibrer la partie de validation et de test, l'inverse est mieux
valid , X_valid , Y_valid = scale_dataset(valid , oversample = False)
test , X_test , Y_test = scale_dataset(test , oversample = False)



"""
    Model for classification:
        K-nearest neighbors:
            We put data on a graph
            And mesure distance between each point with euclidian distance
            d = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    Test:
"""
def knn_modelisation():
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train , Y_train)

    y_pred = knn_model.predict(X_test)
    print(classification_report(Y_test , y_pred))

#Naive Bayes        other model
def Naives_Bayes_Model():
    n_model = GaussianNB()
    n_model = n_model.fit(X_train , Y_train)

    y_pred = n_model.predict(X_test)

    print(classification_report(Y_test, y_pred))

"""
    #Logistic Regression

    ln(P/1 - P) = mx +b

    <==>    P/1-P = e(mx +b)
    <==>    P = (1-P)* e(mx +b)
    <==>    P = 1/(1+e(-(mx+b)))

    Also knw as the sigmoid function
    S(x) = 1/ (1+e(-x))

    In our case x = mx +b
"""

def logistic_modelisation():
    logistic_model = LogisticRegression()
    logistic_model = logistic_model.fit(X_train , Y_train)

    y_pred = logistic_model.predict(X_test)

    print(classification_report(Y_test, y_pred))

"""
    Suport Vector Machines      SVM

    Sur un graphe l'on met toute les donné et les sépare par un trait en fonction de la classe
    On trace une parralèle qui touche chaque premier point de chaque classe et c'ets notre marge

    Dans el cas ou les donné sontdans une seule dimension l'on peux les placer dasn un repere avecd es coordonnées (x , x**2)
    Ce qui permet de Pouvoir mieux les diviser      C'est un kernel trick
"""

def svm_modelisation():

    svm_model = SVC()
    svm_model = svm_model.fit(X_train , Y_train)

    Y_pred = svm_model.predict(X_test)
    print(classification_report(Y_test , Y_pred))


"""
    Neural Networks:
        Fait appel à la backpropagation et au gradient afin de reduire le loss

        La bakcpropagation est une suite de weitgh update 
        L'on calcul le gradient de loss

        Weigth update =    w0.new = w0.old + a*
                                            a est le learning rate

"""


"""
    TensorFlow

    La librairie tensorflow est une librairie qui permet de generer les modeles tout en gérant tout

"""

#Neural net
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def plot_loss(history):
    plt.plot(history.history['loss'],label = 'loss')
    plt.plot(history.history['val_loss'],label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'],label = 'accuracy')
    plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape = (10,)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train,Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
plot_loss(history)
plot_accuracy(history)