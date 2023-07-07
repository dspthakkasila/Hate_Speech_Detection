# Hate_Speech_Detection
# Importing lib
import pandas as pd
import numpy as np
dataset = pd.read_csv("twitter.csv")
dataset
dataset.isnull().sum()
dataset.info()
dataset.describe()
dataset["labels"] = dataset["class"].map({0 :"Hate speech", 
                                         1:"offensive language", 
                                         2:"No Hate or offensive"})
dataset
data = dataset[["tweet","labels"]]
data
import  re
import  nltk
nltk.download("stopwords")
#removal of stop words and  stemming the words
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
#import stemming
stemmer = nltk.SnowballStemmer("english")
# Data Cleaning
def clean_data(text):
    text = str(text).lower()
    text = re.sub('http?://\s+|www\.s+','',text)
    text = re.sub('\[.*?\]','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' %re.escape(string.punctuation),' ',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\v*','',text)
    # Stop words removing
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    # Stemming the text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean_data)
data
x = np.array(data["tweet"])
y = np.array(data["labels"])
x
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
x  = cv.fit_transform(x)
x
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
x_train
# Building out ML Model
from sklearn.tree import DecisionTreeClassifier
dt =  DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
# Confusion Matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
import seaborn as sns
import matplotlib.pyplot as ply
%matplotlib inline
sns.heatmap(cm, annot = True, fmt = ".1f", cmap = "YlGnBu")
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
sample = "Let's unite and kill all the people who are protesting against the government"
sample = clean_data(sample)
sample
data1 = cv.transform([sample]).toarray()
data1
dt.predict(data1)
