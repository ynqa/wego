from gensim.models import word2vec


def main():
    sentences = word2vec.Text8Corpus('text8')
    model = word2vec.Word2Vec(sentences=sentences,
                              sg=0,
                              size=100,
                              window=5,
                              workers=4,
                              negative=5,
                              iter=1,
                              batch_words=10000)

if __name__ == '__main__':
    main()
