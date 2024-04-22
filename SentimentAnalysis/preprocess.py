import re
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def normalization(text):
    # Expand contractions
    expanded_text = contractions.fix(text)

    # make sure all text is lowercase
    expanded_text = expanded_text.lower()

    # # Remove punctuations and special characters
    just_text = re.sub(r'[^a-zA-Z\s]', '', expanded_text)

    # Remove stopwords and trim white space
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(just_text)
    filtered_words = [w for w in word_tokens if not w.lower() in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

if __name__ == "__main__":
    text = "I can't believe it's already 2021. I'm so excited for the new year."
    print(normalization(text))


