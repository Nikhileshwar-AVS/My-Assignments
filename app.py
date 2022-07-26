import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import seaborn as sns
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#!pip install -q scikit-plot
import scikitplot as skplt

# !pip install wordcloud
from wordcloud import WordCloud
from PIL import Image
from textblob import TextBlob


class NLP:
    def __init__(self, data):
        self.data = data


    def stemming(self, column_name):
        '''Cleaning data using re, stemming and stopwords'''
        try:
            corpus = []
            stemming = PorterStemmer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', " ", self.data[column_name][i])
                tweet = re.sub('http', "", tweet)
                tweet = re.sub('co', "", tweet)
                tweet = re.sub('amp', "", tweet)
                tweet = re.sub('new', "", tweet)
                tweet = re.sub('one', "", tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [stemming.stem(word) for word in tweet if word not in set(stopwords.words("english"))]
                tweet = " ".join(tweet)
                corpus.append(tweet)
            
        except Exception as e:
            print("stemming ERROR : ",e)
        
        else:
            
            return corpus


    def lemmatizing(self, column_name):
        '''Cleaning data using re, Lemmatization and stopwords'''
        try:
            corpus = []
            lemmatize = WordNetLemmatizer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', " ", self.data[column_name][i])
                tweet = re.sub('http', "", tweet)
                tweet = re.sub('co', "", tweet)
                tweet = re.sub('amp', "", tweet)
                tweet = re.sub('new', "", tweet)
                tweet = re.sub('one', "", tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatize.lemmatize(word) for word in tweet if word not in set(stopwords.words("english"))]
                tweet = " ".join(tweet)
                corpus.append(tweet)
            
        except Exception as e:
            print("Lemmatizing ERROR : ",e)
    
        else:
            
            return corpus


    def count_vectorizing(self, corpus, max_features = 3000, ngram_range=(1,2)):
        '''Creating Bag of Words using CountVectorizer'''
        try:
            cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = cv.fit_transform(corpus).toarray()
        
        except Exception as e:
            print("count_vectorizing ERROR : ",e)
        
        else:
            
            return X



    def tf_idf(self, corpus, max_features = 3000, ngram_range=(1,2)):
        '''Creating Bag of Words using TfidfVectorizer'''
        try:
            tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = tfidf.fit_transform(corpus).toarray()
        
        except Exception as e:
            print("tf_idf ERROR : ",e)
        
        else:
            
            return X


    def y_encoding(self, target_label):
        """One Hot Encoding if target variable are not in form of 1s and 0s"""
        try:
            y = pd.get_dummies(self.data[target_label], drop_first = True)

        except Exception as e:
            print("y_encoding ERROR : ", e)

        else:
            
            return y
    


    def split_data(self, X, y, test_size = 0.25, random_state = 0):
        '''Splitting data into train and test set'''
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
        
        except Exception as e:
            print("split_data ERROR : ",e)

        else:
            
            return X_train, X_test, y_train, y_test


    def naive_model(self, X_train, X_test, y_train, y_test):
        '''Prediction of model using naive_bayes'''
        try:
            naive = MultinomialNB()
            naive.fit(X_train , y_train)

            y_pred = naive.predict(X_test)
        
        except Exception as e:
            print("naive_model ERROR : ", e)
            
        else:
            
            return y_pred


    def cm_accuracy(self, y_test, y_pred):
        '''Performace Metrics'''
        try:

            skplt.metrics.plot_confusion_matrix(y_test, 
                                                y_pred,
                                                figsize=(15,15))
            plt.savefig('CM.jpg')
            img_cm= Image.open("CM.jpg")
            accuracy = accuracy_score(y_test, y_pred)
        
        except Exception as e:
            print("cm_accuracy ERROR : ", e)

        else:
            return accuracy, img_cm


    def word_cloud(self, corpus):
        '''Generating Word Cloud'''
        try:
            wordcloud = WordCloud(
                        background_color='black',
                        width=1020,
                        height=500,
                        ).generate(" ".join(corpus))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.savefig('WC.jpg')
            img= Image.open("WC.jpg") 
            

        except Exception as e:
            print("word_cloud ERROR : ", e)

        else:
            # print("word cloud plotted")
            return img



    def sentimental_analysis_clean(self, text):
        try:
            text = re.sub('http', "", text)
            text = re.sub('co', "", text)
            text = re.sub('amp', "", text)
            text = re.sub('new', "", text)
            text = re.sub('one', "", text)
            text = re.sub('@[A-Za-z0–9]+', '', text)
            text = re.sub('#', '', text)
            text = re.sub('RT[\s]+', '', text)
            text = re.sub('https?:\/\/\S+', '', text)
        
            return text

        except Exception as e:
            print("sentimental_analysis_clean ERROR : ", e)





# Create space betwwen two context
def space():
    st.markdown("<br>", unsafe_allow_html=True)

##st.markdown("<style>body {background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");background-size: cover;</style>", unsafe_allow_html=True)

# Heading
st.markdown("<h1 style='text-align: center; color: #3f3f44'>NLP - Uber Users Reviews</h1>", unsafe_allow_html=True)
space()
# Sub-Heading
#st.markdown("<strong><p style='color: #424874'>1) This project uses Naive Bayes Algorithm</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>2) You can choose different cleaning process (Stemming, Lemmatizing)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>3) Different type of  Metrics formation (Count Vectorizing, TF-IDF)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>4) Plotting Sentimental Analysis, Confusion Metrics and Word Cloud</p></strong>", unsafe_allow_html=True)
space()



# Preprocessing data
def preprocess():
    try:
        preprocessing_option = ["Stemming", "Lemmatizing"]
        preprocessor = st.selectbox("Select Preprocessing Technique", preprocessing_option)
        return preprocessor
 
    except Exception as e:
        print("preprocess ERROR : ", e)

# hyperparameters
def hyperparameter():
    try:
        features = ["2500", "3000", "3500", "4000"]
        max_features = st.selectbox("Maximum Features you want to restrict?", features)
        space()
        space()
        ranges = ["1,1", "1,2", "1,3"]
        ngram_range = st.selectbox("Combination of words", ranges).split(',')
        return max_features, ngram_range

    except Exception as e:
        print("hyperparameter ERROR : ", e)




# Creating Bag of Words
def boW():
    try:
        metrics = ["count_vectorizing", "tfidf"]
        bag_of_words = st.selectbox("Bag of Word Technique?", metrics)
        return bag_of_words

    except Exception as e:
        print("boW ERROR : ", e)

# Converting Target categorical variable into Numerical Variable
def ylabel():
    try:
        target_variable = ["Yes - If target column does not have values like 0's and 1's", "No"]
        y_option = st.selectbox("Do you want to One Hot Encode Target Variable?", target_variable)
        return y_option

    except Exception as e:
        print("ylabel ERROR : ", e)


# Main function
def main():

    # Uploading data
    df = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    space()
    if df is not None:
        # Reading data
        data = pd.read_csv(df, encoding = "ISO-8859-1")
        st.dataframe(data.head())
        space()

        text = st.selectbox("Select Text Column", data.columns)
        space()
        target = st.selectbox("Select Target Column", data.columns)
        space()
        # print(type(text), type(target))


        # Reassigning feature to DataFrame
        data = data[[text, target]]

        # Droping NaN values
        data = data.dropna()

        # Initialising class "NLP"
        nlp_model = NLP(data)
        
        # Displaying final DataFrame
        st.markdown("<h4 style='color: #438a5e'>Final Dataset</h4>", unsafe_allow_html=True)
        st.dataframe(data.head())
        space()


        # Calling functions for Preprocessing, Bag of Words, Target variable
        preprocessor = preprocess()
        space()
        space()
        max_features, ngram_range = hyperparameter()
        space()
        space()
        bag_of_words = boW()
        space()
        space()
        y_option = ylabel()
        space()
        space()


        # functions
        def metrix(corpus, bag_of_words, max_features, ngram_range):
            try:
                if bag_of_words == "count_vectorizing":
                    X = nlp_model.count_vectorizing(corpus, int(max_features), (int(ngram_range[0]),int(ngram_range[1])))
                    return X

                elif bag_of_words == "tfidf":
                    X = nlp_model.tf_idf(corpus, int(max_features), (int(ngram_range[0]),int(ngram_range[1])))
                    return X

            except Exception as e:
                print("metrix Error : ", e)


        def targetseries(y_option, target):
            try:
                if y_option == "Yes":
                    y = nlp_model.y_encoding(target)
                    return y

                elif y_option == "No":
                    y = data[target]
                    return y
            
            except Exception as e:
                print("targetseries ERROR : ", e)

        # Plotting functions
        def plotwordcloud(corpus, y_test, y_pred):
            st.success("Word Cloud")
            wordcloud = nlp_model.word_cloud(corpus)
            st.image(wordcloud)
            accuracy, cm = nlp_model.cm_accuracy(y_test, y_pred)
            st.success(f"Accuracy : {round(accuracy*100, 2)}%")
            st.image(cm)





        

        # sentiments
        def sentimental(text):
            '''Plotting Sentiments'''

            data['sentiments'] = data[text].apply(nlp_model.sentimental_analysis_clean)

            # Sentiments
            def getSubjectivity(text):
                return TextBlob(text).sentiment.subjectivity

            # Create a function to get the polarity
            def getPolarity(text):
                return  TextBlob(text).sentiment.polarity

            def getAnalysis(score):
                if score < 0:
                    return 'Negative'
                elif score == 0:
                    return 'Neutral'
                else:
                    return 'Positive'


            # Create two new columns 'Subjectivity' & 'Polarity'
            data['Subjectivity'] = data['sentiments'].apply(getSubjectivity)
            data['Polarity'] = data['sentiments'].apply(getPolarity)
            data['Analysis'] = data['Polarity'].apply(getAnalysis)

            st.success("Sentiments")

            sns.countplot(x=data["Subjectivity"],data=data)
            
            st.pyplot(use_container_width=True)
            
            #.countplot(x=data["Analysis"],data=data)


            
                

        # Model Creation
        if st.button("Analyze"):
            space()
            if preprocessor == "Stemming":
                corpus = nlp_model.stemming(text)
                X = metrix(corpus, bag_of_words, max_features, ngram_range)
                y = targetseries(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_model(X_train, X_test, y_train, y_test)
                sentimental(text)
                plotwordcloud(corpus, y_test, y_pred)
                




            elif preprocessor == "Lemmatizing":
                corpus = nlp_model.lemmatizing(text)
                X = metrix(corpus, bag_of_words, max_features, ngram_range)
                y = targetseries(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_model(X_train, X_test, y_train, y_test)
                sentimental(text)
                plotwordcloud(corpus, y_test, y_pred)
                


    #st.markdown("<h4 style='text-align: center; color: #3f3f44'>@2022 Streamlit Application</h4>", unsafe_allow_html=True)

        

 
if __name__ == "__main__":
    main()
