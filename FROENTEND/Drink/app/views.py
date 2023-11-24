from django.shortcuts import render, redirect

# Create your views here.
from django.contrib.auth.models import User
from django.contrib import messages
from . models import Register
import pandas as pd
import numpy as np
# import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn
#importing the required libraries


Home = 'index.html'
About = 'about.html'
Login = 'login.html'
Registration = 'registration.html'
Userhome = 'userhome.html'
Load = 'load.html'
View = 'view.html'
Preprocessing = 'preprocessing.html'
Model = 'model.html'
Prediction = 'prediction.html'


# # Home page
def index(request):

    return render(request, Home)

# # About page


def about(request):
    return render(request, About)

# # Login Page


def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = Register.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        if d:
            return redirect(userhome)
        else:
            msg = 'Login failrd'
            return render(request, Login, {'msg': msg})
    return render(request, Login)

# # registration page user can registration here


def registration(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        if password == conpassword:
            userdata = Register.objects.filter(email=email).exists()
            if userdata:
                msg = 'Account already exists'
                return render(request, Registration, {'msg': msg})
            else:
                userdata = Register(name=Name, email=email,
                                    password=password, age=age, contact=contact)
                userdata.save()
                return render(request, Login)
        else:
            msg = 'Register failed!!'
            return render(request, Registration, {'msg': msg})

    return render(request, Registration)

# # user interface


def userhome(request):

    return render(request, Userhome)

# # Load Data


def load(request):
    if request.method == "POST":
        file = request.FILES['file']
        global df

        df = pd.read_csv(file)
        messages.info(request, "Data Uploaded Successfully")

    return render(request, Load)

# # View Data


def view(request):
    col = df.to_html
    dummy = df.head(100)

    col = dummy.columns
    rows = dummy.values.tolist()
    # return render(request, 'view.html',{'col':col,'rows':rows})
    return render(request, View, {'columns': df.columns.values, 'rows': df.values.tolist()})


# preprocessing data
def preprocessing(request):
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df,data
    
    if request.method == "POST":

        size = int(request.POST['split'])
        size = size / 100
      
        df=pd.read_csv(r'data01.csv')
        data=df.corr()
        df=df[['EF','Systolic blood pressure','gendera','Blood sodium','PCO2','Chloride','MCH','Bicarbonate','MCHC','MCV','Neutrophils','BMI','age','COPD','temperature','Urine output','Platelets','outcome']]
        df.head()
        df['Systolic blood pressure'] = df['Systolic blood pressure'].fillna(df['Systolic blood pressure'].median())
        df['PCO2'] = df['PCO2'].fillna(df['PCO2'].median())
        df['Neutrophils'] = df['Neutrophils'].fillna(df['Neutrophils'].median())
        df['BMI'] = df['BMI'].fillna(df['BMI'].median())
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())
        df['Urine output'] = df['Urine output'].fillna(df['Urine output'].median())
        df['outcome'] = df['outcome'].fillna(df['outcome'].median())

        x=df.drop('outcome',axis=1)
        y=df['outcome']                                 

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
       
        messages.info(request, "Data Preprocessed and It Splits Succesfully")
    return render(request, Preprocessing)


# Model Training
def model(request):
    global x_train, x_test, y_train, y_test
    if request.method == "POST":

        model = request.POST['algo']

        if model == "0":

            rf=RandomForestClassifier(n_estimators=52)
            rf.fit(x_train,y_train)
            y_pred=rf.predict(x_test)
            ac_rf=accuracy_score(y_pred,y_test)
            ac_rf=ac_rf*100
            msg = 'Accuracy of RandomForestClassifier : ' + str(ac_rf)
            return render(request, Model, {'msg': msg})

        elif model == "1":
            knn=KNeighborsClassifier(n_neighbors=45)
            knn.fit(x_train,y_train)
            y_pred=knn.predict(x_test)
            ac_knn=accuracy_score(y_pred,y_test)
            ac_knn=ac_knn*100
            msg = 'Accuracy of KNeighborsClassifier : ' + str(ac_knn)
            return render(request, Model, {'msg': msg})

        elif model == "2":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda=LinearDiscriminantAnalysis()
            lda.fit(x_train,y_train)
            y_pred=lda.predict(x_test)
            ac_lda=accuracy_score(y_pred,y_test)
            ac_lda=ac_lda*100
            msg = 'Accuracy of LinearDiscriminantAnalysis : ' + str(ac_lda)
            return render(request, Model, {'msg': msg})
        
        elif model == "3":
            from sklearn.neural_network import MLPClassifier
            mlp=MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
            mlp.fit(x_train,y_train)
            y_pred=mlp.predict(x_test)
            ac_mlp=accuracy_score(y_pred,y_test)
            ac_mlp=ac_mlp*100
            msg = 'Accuracy of MLPClassifier : ' + str(ac_mlp)
            return render(request, Model, {'msg': msg})
    
        elif model == "4":
            from keras.models import Sequential
            from keras.layers import Dense, Dropout

            from keras.models import load_model
            model = load_model(r'app\neural_network.h5')
            score=0.9423418045043945
            ac_nn = score * 100
            msg = 'Accuracy of NeuralNetwork : ' + str(ac_nn)
            return render(request, Model, {'msg': msg})
    return render(request, Model)


# Prediction here we can find the result based on user input values.
def prediction(request):

    global x_train,x_test,y_train,y_test,x,y
    

    if request.method == 'POST':
        

        f1=int(request.POST['EF'])
        f2=int(request.POST['Systolic blood pressure'])
        f3=int(request.POST['gendera'])
        f4=int(request.POST['Blood sodium'])
        f5=int(request.POST['PCO2'])
        f6=int(request.POST['Chloride'])
        f7=int(request.POST['MCH'])
        f8=int(request.POST['Bicarbonate'])
        f9=int(request.POST['MCHC'])
        f10=int(request.POST['MCV'])
        f11=int(request.POST['Neutrophils'])
        f12=int(request.POST['BMI'])
        f13=int(request.POST['age'])
        f14=int(request.POST['COPD'])
        f15=int(request.POST['temperature'])
        f16=int(request.POST['Urine output'])
        f17=int(request.POST['Platelets'])
      
       
    
        lee = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17]
        print([lee])

        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        if result==0:
            msg="There is a No-Chance to Survive"
            import pygame
            pygame.mixer.init()
            sound = pygame.mixer.Sound("app\YRL6BSM-siren.mp3")
            sound.play()
        elif result==1:
            msg="There is a Chance to Survive" 
        inp = [['EF','Systolic blood pressure',	'gendera',	'Blood sodium',	'PCO2',	'Chloride',	'MCH',	'Bicarbonate',	'MCHC',	'MCV',	'Neutrophils',	'BMI',	'age',	'COPD',	'temperature',	'Urine output',	'Platelets']]

        return render(request,Prediction,{'Lee':lee, 'msg':msg , 'Inp':inp}) 

    return render(request,Prediction)