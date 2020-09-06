from gensim.models import KeyedVectors
def word2vec():
    Embedding_model =KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin',binary=True)
    print("Embedding　OK")
    return Embedding_model
