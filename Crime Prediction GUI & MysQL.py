###################################################################################################################################
#GUI and MySQL
from tkinter import *
import tkinter.messagebox
import mysql.connector
#import seaborn as sns
from PIL import Image,ImageTk

#from nltk.stem import *
#Stemming is the process of producing morphological variants of a root/base word.
#A stemming algorithm reduces the words “chocolates”, “chocolatey”, “choco” to the root word “chocolate”.

def MAIN():
  R1=Tk()
  R1.geometry('800x600')
  R1.title('WELCOME-1')

  image=Image.open('c1.jpg')
  image=image.resize((900,600))
  photo_image=ImageTk.PhotoImage(image)
  l=Label(R1, image=photo_image)
  l.place(x=0, y=0)
    
  l=Label(R1, text="WELCOME TO CRIME PREDICTION PORTAL",font=('algerain',14), bg="lightblue")
  l.place(x=180, y=30)

  b1=Button(R1, text="Register",width=10,height=2,font=('algerain',14), bg="lightblue", fg="red", command=m1)
  b1.place(x=250, y=400)
  
  b2=Button(R1, text="Login",width=10,height=2, font=('algerain',14), bg="lightblue", fg="red", command=m3)
  b2.place(x=420, y=400)
  
  R1.mainloop()


def m1():
  def m2():
    username=e1.get()
    password=e2.get()
    email=e3.get()
    phoneno=e4.get()

    a=mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", database="crime")
    b=a.cursor()
    b.execute("INSERT INTO t1 VALUES(%s,%s,%s,%s)",(username,password,email,phoneno))
    a.commit()

    if e1.get()=="" or e2.get=="":
      tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
    else:
      tkinter.messagebox.showinfo("WELCOME %s" %username, "Lets Login")
      m3()
    
  R2=Toplevel()
  R2.geometry('600x500')
  R2.title('Register and Login')

  image=Image.open('register.jpg')
  image=image.resize((900,700))
  photo_image=ImageTk.PhotoImage(image)
  l=Label(R2, image=photo_image)
  l.place(x=0, y=0)

  #l=Label(R2, text="Login & Register", font=('algerain',14,'bold'), fg="orange")
  #l.place(x=200, y=50)

  l1=Label(R2, text="Username", font=('algerain',14),bg="lightblue", fg="black")
  l1.place(x=100, y=200)
  l2=Label(R2, text="Password", font=('algerain',14),bg="lightblue", fg="black")
  l2.place(x=100, y=250)
  l3=Label(R2, text="Email", font=('algerain',14),bg="lightblue", fg="black")
  l3.place(x=100, y=300)
  l4=Label(R2, text="Phoneno", font=('algerain',14),bg="lightblue", fg="black")
  l4.place(x=100, y=350)
  
  e1=Entry(R2, font=14)
  e1.place(x=200, y=205)
  e2=Entry(R2, font=14, show="**")
  e2.place(x=200, y=255)
  e3=Entry(R2, font=14)
  e3.place(x=200, y=305)
  e4=Entry(R2, font=14)
  e4.place(x=200, y=355)

  b1=Button(R2, text="Signup",width=8,height=1, font=('algerain',14), bg="lightblue", fg="red", command=m2)
  b1.place(x=250, y=400)
      
  R2.mainloop()

def m3():
    def m4():
        a=mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", database="crime")
        b=a.cursor()
        username=e1.get()
        password=e2.get()

        if (e1.get()=="" or e2.get()==""):
            tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
        else:
            b.execute("SELECT * FROM t1 WHERE username=%s AND password=%s",(username,password))

            if b.fetchall():
                tkinter.messagebox.showinfo("WELCOME %s" % username, "Logged in successfully")
                m5()#from function def m5() Function call
                
            else:
                tkinter.messagebox .showinfo("Sorry", "Wrong Password")
                    
    R3=Toplevel()
    R3.geometry('600x500')
    R3.title('Login')
    #R3.configure(background='lightblue')

    image=Image.open('login.jpg')
    image=image.resize((900,700))
    photo_image=ImageTk.PhotoImage(image)
    l=Label(R3, image=photo_image)
    l.place(x=0, y=0)

    #l=Label(R3, text="Login", font=('algerain',14,'bold'), fg="orange")
    #l.place(x=200, y=50)

    l1=Label(R3, text="Username", font=('algerain',14), bg="lightblue",fg="black")
    l1.place(x=100, y=200)
    l2=Label(R3, text="Password", font=('algerain',14), bg="lightblue", fg="black")
    l2.place(x=100, y=250)
      
    e1=Entry(R3, font=14)
    e1.place(x=200, y=205)
    e2=Entry(R3, font=14, show="**")
    e2.place(x=200, y=255)

    b1=Button(R3, text="Login",width=8,height=1, font=('algerain',14),bg="lightblue", fg="red", command=m4)
    b1.place(x=250, y=400)

    R3.mainloop()

def m5():
  R1=Toplevel()
  R1.geometry('800x600')
  R1.title('WELCOME-2')

  image=Image.open('c2.jpg')
  image=image.resize((900,600))
  photo_image=ImageTk.PhotoImage(image)
  l=Label(R1, image=photo_image)
  l.place(x=0, y=0)
  
  l=Label(R1, text="Crime Prediction", font=('algerain',18,'bold'))
  l.place(x=220, y=100)

  b1=Button(R1, text="Naive Bayes",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=NaiveBayes)
  b1.place(x=250, y=200)
  
  b4=Button(R1, text="Random Forest",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=RF)
  b4.place(x=250, y=300)

  b2=Button(R1, text="Pie-Chart",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Pie_Chart)
  b2.place(x=250, y=400)

  b2=Button(R1, text="Prediction",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Prediction)
  b2.place(x=250, y=500)
  
  R1.mainloop()

  
#Data Preprocessing
def NaiveBayes():
  print('\n\n------------------Data Preprocessing------------------------------\n')

  import numpy as np    #numpy library used for array and matrix computation
  import pandas as pd   #pandas library used for data manipulation and analysis
  import warnings       #warnings module used to remove warnings
  warnings.filterwarnings('ignore')

  # Importing the dataset
  dataset = pd.read_csv('dataset.csv',encoding='latin1')
  #print(dataset)

  #Convert to factorize the dataset
  #pandas.factorize() method helps to get the numeric representation of an array by identifying distinct values
  dataset['city'] = pd.factorize(dataset['city'])[0]
  dataset['crime'] = pd.factorize(dataset['crime'])[0]
  dataset['incidences'] = pd.factorize(dataset['incidences'])[0]

  #Split the X & Y variables
  X=dataset.drop(['city','year','incidences','rate'],axis=1)
  Y=dataset['crime']

  #X = dataset.iloc[:, [2,3,28,34]].values

  #Encoding Y variable
  from sklearn.preprocessing import LabelEncoder
  labelencoder_y= LabelEncoder()
  #Y=Y.shape
  #Y=Y.reshape(-1,1)
  Y = labelencoder_y.fit_transform(Y)
  #print(Y)

  # Splitting the dataset into the Training set and Test set
  #Scikit-learn/Sklearn is an open source Python library that has powerful tools for data analysis and data mining. Ex: Classification,Regression,Clustering etc.
  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
  #print(X_train)
  #print(X_test)
  #print(Y_train)
  #print(Y_test)

  # Feature Scaling
  ####Feature Scaling Of Datasets
  #create the object of StandardScaler class for independent variables or features
  #Feature Scaling: is a technique to standardize the independent variables of the dataset in a specific range.
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  #1.fit(): is a method of calculating that mean value(of column). When we are training a  model then we use the fit() method on your training dataset.
  #2.transform(): is a method will just replaces the NaNs(Not a Number) in the column with the newly calculated value, and returns the new dataset. When we are training a  model then we use the transform() method on your test dataset.
  #3.fit_transform(): is a method which performs both the fit() and transform() methods internally.
  #Hence fit_transform replaces calculated mean value in the NAN columns.
  X_train = sc.fit_transform(X_train)
  #print(X_train)   #It gives matrix output
  X_test = sc.transform(X_test)
  #print(X_test)    #It gives matrix output

  print('\n\n-------------------Naive Bayes algorithm-----------------------')

  #Naive Bayes algorithm is a supervised machine learning classification algorithm which is based on Bayes Theorem to predict the output.
  from sklearn.naive_bayes import GaussianNB
  classifier = GaussianNB()
  classifier.fit(X_train, Y_train)

  # Predicting the Test set results
  Y_pred = classifier.predict(X_test)
  #print(Y_pred)

  #Print the accuracy
  #Accuracy is defined as the percentage of correct predictions for the test data

  from sklearn.metrics import accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  A = accuracy_score(Y_test,Y_pred)
  print('Accuracy Score: {}\n'.format(A))

def RF():
  print('\n\n------------------Data Preprocessing------------------------------\n')

  import numpy as np    #numpy library used for array and matrix computation
  import pandas as pd   #pandas library used for data manipulation and analysis
  import warnings       #warnings module used to remove warnings
  warnings.filterwarnings('ignore')

  # Importing the dataset
  dataset = pd.read_csv('dataset.csv',encoding='latin1')
  #print(dataset)

  #Convert to factorize the dataset
  #pd.factorize() is useful for obtaining a numeric representation
  dataset['city'] = pd.factorize(dataset['city'])[0]
  #print(dataset)
  dataset['crime'] = pd.factorize(dataset['crime'])[0]
  dataset['incidences'] = pd.factorize(dataset['incidences'])[0]

  #Split the X & Y variables
  X=dataset.drop(['city','year','incidences','rate'],axis=1)
  Y=dataset['crime']

  # Splitting the dataset into the Training set and Test set
  #Scikit-learn/Sklearn is an open source Python library that has powerful tools for data analysis and data mining. Ex: Classification,Regression,Clustering etc.
  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

  # Feature Scaling
  ####Feature Scaling Of Datasets
  #create the object of StandardScaler class for independent variables or features
  #Feature Scaling: is a technique to standardize the independent variables of the dataset in a specific range.
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  #1.fit(): is a method of calculating that mean value(of column). When we are training a  model then we use the fit() method on your training dataset.
  #2.transform(): is a method will just replaces the NaNs(Not a Number) in the column with the newly calculated value, and returns the new dataset. When we are training a  model then we use the transform() method on your test dataset.
  #3.fit_transform(): is a method which performs both the fit() and transform() methods internally.
  #Hence fit_transform replaces calculated mean value in the NAN columns.
  X_train = sc.fit_transform(X_train)
  #print(X_train)   #It gives matrix output
  X_test = sc.transform(X_test)
  #print(X_test)    #It gives matrix output

  print('\n\n-------------------RandomForest algorithm-----------------------')

  #RandomForest algorithm is a supervised classification machine learning algorithm which is an ensemble of decision trees for the prediction.
  from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier()
  classifier.fit(X_train, Y_train)

  # Predicting the Test set results
  Y_pred = classifier.predict(X_test)
  #print(Y_pred)

  ##Print the  accuracy
  #Accuracy is the fraction of predictions our model got right
  #Scikit-learn is a free software machine learning library for the Python programming language.
  #It features various classification, regression and clustering algorithms including support vector machines
  from sklearn.metrics import accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  A = accuracy_score(Y_test,Y_pred)
  print('Accuracy Score: {}\n'.format(A))






#Pie_Chart
def Pie_Chart():
    print('\n------------Plotting Pie_Chart------------')
    import pandas as pd
    #import seaborn as sns
    import matplotlib.pyplot as plt
        
    data = pd.read_csv('dataset.csv')
    d = data.groupby('city')[['rate']].sum()
    
    #%1.2f%% returns value 25.34%, and .1f% returns 25.3%
    plt.pie(d['rate'], labels = d.index,autopct = '%1.2f%%')
    plt.show()


#Prediction
def Prediction():
  import numpy as np  
  import matplotlib.pyplot as plt
  import pandas as pd

  # Importing the dataset
  dataset = pd.read_csv('dataset.csv')

  #Convert to factorize the dataset
  #pd.factorize() is useful for obtaining a numeric representation
  dataset['city'] = pd.factorize(dataset['city'])[0]
  dataset['incidences'] = pd.factorize(dataset['incidences'])[0]
  #print('city',dataset)

  #Split the X & Y variable
  X = dataset.iloc[:, [0,1,3,4]].values
  y = dataset.iloc[:,2].values

  # Splitting the dataset into the Training set and Test set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  # Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  #X_train = pd.get_dummies(0,1)
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # Fitting RF to the Training set
  from sklearn.naive_bayes import GaussianNB
  classifier = GaussianNB()
  classifier.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = classifier.predict(X_test)


  #Enter the City and Year
  city = int(input('\nenter a city code:'))
  if city in range(0,2500):
      
      year = int(input('enter an year:'))
      if year in range(2021,2026):

          incidences = int(input('enter the incidences:'))
          if incidences in range(0,10000):

              rate = int(input('enter the rate:'))
              if rate in range(0,200):
                  
                  z = classifier.predict([[city,year,incidences,rate]])
                  print('\nThe possibility of the crime is:\n',z)

              else:
                  print('Incorrect rate details-please enter the correctly!!')
                  
          else:
              print('Incorrect incidences details-please enter the correctly!!')   
    
      else:
          print('Incorrect Year details-please enter the correctly!!')   
  else:
      print('Incorrect city code details-please enter the correctly!!')
       
MAIN()
###################################################################################################################################
