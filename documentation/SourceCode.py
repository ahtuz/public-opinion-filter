import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles (path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filesname, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filesname)

    return DataFrame(rows, index=index)

data = DataFrame ({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\B21\Semester 3\AI\public-opinion-filter\spam', 'spam'),sort=True)
data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\B21\Semester 3\AI\public-opinion-filter\ham', 'ham' ),sort=True) 

print("adsfghj")

data.head()

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB
targets =  data['class'].values
classifier.fit(counts, targets)

examples = ['asdfghjkl']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions

print(predictions)