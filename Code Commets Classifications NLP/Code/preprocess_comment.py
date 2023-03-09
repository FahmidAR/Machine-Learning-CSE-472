import re
import string

from scipy.io import arff
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# # pip install liac-arff


# Load the CSV file
df = pd.read_csv('java.csv')

# Define the text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token in stop_words]
    # Stemming
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens into a string
    text = ' '.join(tokens)
    return text

# Apply the preprocessing function to the 'comment_sentence' column
df['comment_sentence'] = df['comment_sentence'].apply(preprocess_text)

#take last collum named category and divide into csv according to its unuiqe value


# Save the preprocessed data to a new CSV file
df.to_csv('java_2.csv', index=False)


# # Save the preprocessed DataFrame to an ARFF file
# with open('java_2.arff', 'w') as f:
#     arff.dump({'data': df.values.tolist(), 'attributes': list(df.columns)}, f, relation='comments')

# # Convert the DataFrame to an ARFF string
# arff_data = arff.arff_creator.create_attribute_string(df.columns.tolist()) + '\n' + '\n'.join(arff.arff_creator.create_data_string(df))

# # Write the ARFF string to a file
# with open('java_2.arff', 'w') as f:
#     f.write(arff_data)