import fasttext
import gensim.downloader as api

class FastTextEncoder():
    """
    A wrapper for FastText encoder, which is used to encode text into vectors.
    This class is used to encode the text extracted from nodes of a scene graph into vectors, which are then used as input for the conditioning encoder network.
    """
    model = fasttext.load_model('models/cc.en.300.bin')

    @staticmethod
    def encode(text):
        """
        Input: text [str]
        Output: text_embedd [1, 300]
        This method encodes the text into a vector of size 300.
        """
        return FastTextEncoder.model.get_word_vector(text).reshape(1, -1)
    
class Word2VecEncoder():
    """
    A wrapper for Word2Vec encoder, which is used to encode text into vectors.
    This class is used to encode the text extracted from nodes of a scene graph into vectors, which are then used as input for the conditioning encoder network.
    """
    model = api.load('word2vec-google-news-300')
    
    @staticmethod
    def encode(word):
        """
        Input: word (a single word) [str]
        Output: text_embedd [1, 300]
        This method encodes the text into a vector of size 300.
        """
        word = word.lower()
        try:
            word_embedd = Word2VecEncoder.model[word]
        except KeyError:
            word_embedd = Word2VecEncoder.model['unk']
        word_embedd = word_embedd.reshape(1, -1)
        return word_embedd

if __name__ == '__main__':
    word = 'apple'
    word_embedd = FastTextEncoder.encode(word)
    print(word_embedd.shape)
    print(word_embedd)