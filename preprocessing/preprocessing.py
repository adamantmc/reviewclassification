import datetime
import csv
import os
import json

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from os import listdir

'''
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

#Stop word list
en_stop = get_stop_words('en')
#Porter Stemmer
p_stemmer = PorterStemmer()

review = "I found this movie to be a big disappointment, especially considering the cast. The characters are not believable, as are the ridiculous circumstances in which they find themselves. The only part of the film I enjoyed was when the most annoying characters finally get killed. The special effects consist mostly of scenes of gory dead or dying bodies. A typical unimaginative slasher flick. It's hard to believe, make that impossible to believe that a reclusive creature that sneaks up on goats in the middle of the night could be captured by a group of clumsy, noisy idiots. Equally impossible to believe is how they knew exactly were to find it, in spite of the fact the creature has evaded capture, or even photographing. The man that pulls off the impossible in capturing the Chupacabra alive is our one dimensional Dr. Pena (Giancarlo Esposito). The only thing Dr. Pena is more obsessed with than the creature is his dart gun. A dart gun that works were mere bullets fail. The captain of the ship (John Rhys-Davies) is introduced as a 'war veteran'. He employs his military prowess by having his men shoot at the creature, regardless of were on the ship they happen to be. The Navy Seals that show up from nowhere repeat the pattern of shooting at everything. Dylan Neal plays an insurance investigator brought on board the cruise ship to catch a thief. He spends most of the movie tagging along with whomever is trying to kill the creature at the moment. The creature doesn't even closely resemble a Chupacabra. It doesn't behave like one either. Instead of a small, shy, secretive animal that hunts by stealth at night, we get a bulletproof Freddy Kruger, killing everything in sight. A simple search on Google would have been very helpful to the writers and the special effects crew."

words = [i for i in tokens if not i in en_stop]
review_2 = ''.join((w+" ") for w in words)

print(review_1)
print()
print(review_2)
'''

def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def review_to_JSON(review):
    json_review = {}
    json_review['text'] = review[1]
    json_review['pos_score'] = review[2]
    json_review['neg_score'] = review[3]
    json_string = json.dumps(json_review)
    return json_string

def getSentimentDictionary(filename):
    sentiment_dict = dict()
    with open(filename) as file:
        for row in file:
            if row.startswith("a") or row.startswith("v"):
                temp_list = row.split()[0:4]
                word = row.split()[4].split("#")[0]
                if word not in sentiment_dict:
                    sentiment_dict[word] = [float(row.split()[2]), float(row.split()[3])]
    return sentiment_dict

def process_set(paths):
    reviews = []
    for path in paths:
        fileset = [f for f in listdir(path)]
        for f in fileset:
            with open(path+"/"+f, 'r') as content_file:
                content = content_file.read()
            soup = BeautifulSoup(content, 'html.parser')
            tokens = word_tokenize(soup.getText())
            tagged_tokens = pos_tag(tokens)
            reviews.append(("processed_"+path+"/"+f.split('.')[0]+".json", ''.join((x+" ") for (x,y) in tagged_tokens if y.startswith("JJ") or y.startswith("V") ), 0, 0))
            #reviews.append(("processed_"+path+"/"+f.split('.')[0]+".json", soup.getText(), 0, 0))
    return reviews

tlog("Reading reviews and removing html elements.")
reviews = process_set(("data/train/pos", "data/train/neg"))
tlog("Reviews processed.")

tlog("Reading sentiment dictionary.")
sent_dict = getSentimentDictionary("SentiWordNet.txt")
tlog("Sentiment dictionary read.")

tp = 0
fn = 0

tlog("Calculating scores.")

#Create processed directories if they don't exist
make_dir("processed_data")
make_dir("processed_data/train")
make_dir("processed_data/test")
make_dir("processed_data/train/pos")
make_dir("processed_data/train/neg")

for review in reviews:
    pos = 0
    neg = 0

    for word in review[1].split():
        scores = sent_dict.get(word, [0,0])
        pos += scores[0]
        neg += scores[1]

    review = (review[0], review[1], pos, neg)

    f = open(review[0], 'w')
    f.write(review_to_JSON(review))
    f.close()

tlog("Done.")

