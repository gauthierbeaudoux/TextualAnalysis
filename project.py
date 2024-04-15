import nltk
from nltk.corpus import stopwords
import spacy
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')


with open('discoursGreeningFinance.txt', 'r') as file:
    real_text = file.read()

print(len(real_text))

text = real_text
# print(text)

# Subsitution and cleaning
text = text.replace(',','')
text = text.replace(':','')
text = text.replace(';','')
text = text.replace('-','')
text = text.lower()
# print(text)
print(len(text))


# Tokenization
print("\nTokenization")
tokens = nltk.word_tokenize(text)
print(tokens)
print(f"{len(tokens) = }")


# Stop word
print("\nStop word")
stop_words = set(stopwords.words('english'))
print(stop_words)
print()
token_stop_words = [word for word in tokens if word not in stop_words]
print(token_stop_words)
print(f"{len(token_stop_words) = }")


# Lemmatization
print("\nLemmatization")
nlp = spacy.load('en_core_web_sm')
text_filtrated = " ".join(token_stop_words)
lemmatized_tokens = [token.lemma_ for token in nlp(text_filtrated)]
print(f"{lemmatized_tokens = }")


lemmatization_string = " ".join(lemmatized_tokens)


# WordCloud
# wordcloud = WordCloud(background_color = 'white', max_words = 50).generate(lemmatization_string)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

# wordcloud = WordCloud(background_color = 'white', max_words = 50).generate(real_text)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()




# Blob



blob = TextBlob(real_text)
# print("\nTags :")
# print(blob.tags)

print("\nNoun Phrases")
print(blob.noun_phrases)

print("\nSentences")
print(blob.sentences)

print("\nPolarity and objectivity of the sentences")
print(blob.sentiment)

print(len(blob.sentences))
print()


for sentence in blob.sentences:
    polarite, subjectivite = sentence.sentiment
    if polarite > 0.4:
        print(f"Phrase : {sentence}")
        print(sentence.sentiment)




