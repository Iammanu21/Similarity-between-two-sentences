import pandas as pd
import numpy as np
import re





#Importing Dataset
dataset = pd.read_csv(r'input.csv', quotechar = '"' ,engine='python',skipinitialspace=True)
corpus1 = []
corpus2 =[]





# Importing nltk libraby and dependncy
# Preprocessing the data and removing the stopwords
# Saving the pre-processed data in corpus
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()






# Out of 3 lack observation, took only 9000 for faster training of the model.
# Accuracy could increase if more data was taken
for i in range(0,9000):
  Q1 = re.sub('[^a-zA-Z]',' ',dataset['Question 1'][i])
  Q1 = Q1.lower()
  Q1 = Q1.split()
  Q1 = [ps.stem(word) for word in Q1 if not word in set(stopwords.words('english'))]
  Q1=' '.join(Q1)
  corpus1.append([Q1])
  
for j in range(0,9000):
  Q2 = re.sub('[^a-zA-Z]',' ',dataset['Question 2'][j])
  Q2 = Q2.lower()
  Q2 = Q2.split()
  Q2 = [ps.stem(word) for word in Q2 if not word in set(stopwords.words('english'))]
  Q2=' '.join(Q2)
  corpus2.append([Q2])
# Converting the corpuses into arrays and concatenating  
corpus1 = np.asarray(corpus1)
corpus2 = np.asarray(corpus2)
corpus = np.concatenate((corpus1,corpus2),axis=1)






# Defining the function to compute the cosine similarity/Changing the texts to vectors
import math
from collections import Counter
WORD = re.compile(r'\w+')

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator




 
# Saving the cosine similarities in an array    
cos_dis = []
for i in range(0,9000):
        text1 = corpus[i][0]
        text2 = corpus[i][1]
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
        cosine = get_cosine(vector1, vector2)
        cos_dis.append([cosine])
cos_dis = np.asarray(cos_dis)        
Y = dataset.iloc[0:9000,2:3]

df= pd.DataFrame(cos_dis)   
df.to_csv('output.csv')


# Splitting the dataset in training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cos_dis, Y, test_size=0.2)






# Using random forest classifer for classfication problem
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion= 'entropy')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)






# Calculating the accuracy
from sklearn.metrics import accuracy_score as ac
score = ac(y_test,y_pred)


  