import os
from pyspark import SparkContext

sc = SparkContext()

def filter_stop_words(word):
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words("english")
    return word not in english_stop_words

def load_text(text_name):
    current_directory = os.path.dirname(__file__)

    # Split text in words
    # Remove empty word artefacts
    # Remove stop words ('I', 'you', 'a', 'the', ...)
    vocabulary = sc.textFile(os.path.join(current_directory, text_name))\
        .flatMap(lambda lines: lines.lower().split())\
        .flatMap(lambda word: word.split("."))\
        .flatMap(lambda word: word.split(","))\
        .flatMap(lambda word: word.split("!"))\
        .flatMap(lambda word: word.split("?"))\
        .flatMap(lambda word: word.split("'"))\
        .flatMap(lambda word: word.split("\""))\
        .filter(lambda word: word is not None and len(word) > 0)\
        .filter(filter_stop_words)

    # Count the total number of words in the text
    word_count = vocabulary.count()

    # Compute the frequency of each word: frequency = #appearances/#word_count
    word_freq = vocabulary.map(lambda word: (word, 1))\
        .reduceByKey(lambda count1, count2: count1 + count2)\
        .map(lambda (word, count): (word, count/float(word_count)))\

    return word_freq

def main():
    iliad = load_text('iliad.mb.txt')
    odyssey = load_text('odyssey.mb.txt')

    # Join the two datasets and compute the difference in frequency
    # Note that we need to write (freq or 0) because some words do not appear
    # in one of the two books. Thus, some frequencies are equal to None after
    # the full outer join.
    join_words = iliad.fullOuterJoin(odyssey)\
        .map(lambda (word, (freq1, freq2)): (word, (freq2 or 0) - (freq1 or 0)))

    # 10 words that get a boost in frequency in the sequel
    emerging_words = join_words.takeOrdered(10, lambda (word, freq_diff): -freq_diff)
    # 10 words that get a decrease in frequency in the sequel
    disappearing_words = join_words.takeOrdered(10, lambda (word, freq_diff): freq_diff)

    # Print results
    for word, freq_diff in emerging_words:
        print "%.2f" % (freq_diff*10000), word
    for word, freq_diff in disappearing_words[::-1]:
        print "%.2f" % (freq_diff*10000), word

    raw_input("press ctrl+c to exit")

if __name__ == "__main__":
    main()
