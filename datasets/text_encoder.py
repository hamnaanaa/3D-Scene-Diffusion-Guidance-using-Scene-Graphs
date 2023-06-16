import fasttext

class FastTextEncoder():
    """
    A wrapper for FastText encoder, which is used to encode words into vectors.
    This class is used to encode the text extracted from nodes of a scene graph into vectors, which are then used as input for the conditioning encoder network.
    """
    def __init__(self):
        self.model = fasttext.load_model('models/cc.en.300.bin')

    def encode(self, word):
        """
        Input: word [str]
        Output: word_embedd [1, 300]
        This method encodes the word into a vector of size 300.
        """
        return self.model.get_word_vector(word).reshape(1, -1)
  
