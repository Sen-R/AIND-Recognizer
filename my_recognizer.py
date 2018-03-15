import warnings
from asl_data import SinglesData

def get_model_ll_safe(model, X, lengths):
    """ Returns model.score(X, lengths) unless an exception is thrown,
    in which case returns float("-inf")
    """
    try:
        return model.score(X, lengths)
    except:
        return float("-inf")

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = [{k:get_model_ll_safe(model, X, lengths) for k, model in models.items()}
                     for X, lengths in test_set.get_all_Xlengths().values()]
    guesses = [max(prob, key=prob.get) for prob in probabilities]
    # TODO implement the recognizer
    return probabilities, guesses
    
