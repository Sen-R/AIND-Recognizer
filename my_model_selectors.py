import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        best_bic=float("inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                if self.verbose:
                    print("Trying model for {} with {} states".format(self.this_word, n))
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                this_score = hmm_model.score(self.X, self.lengths)
            except:
                if self.verbose:
                    print("BIC model creation failure on {} with {} states".format(self.this_word, n))
                continue
            this_bic = -2*this_score + np.log(len(self.X))*self.count_model_parameters(hmm_model)
            if self.verbose:
                print("LL={:0.2f}, n_samp={}, n_params={}, BIC={:0.2f}".format(this_score,
                                                                               len(self.X),
                                                                               self.count_model_parameters(hmm_model),
                                                                               this_bic))
            if this_bic < best_bic:
                best_bic = this_bic
                best_model = hmm_model
                best_n = n

        if self.verbose:
            if best_model is None:
                print("BIC couldn't create any model within given range of n_components")
            else:
                print("Best model for {} has {} states".format(self.this_word, best_n))
        return best_model
        

    def count_model_parameters(self, hmm_model):
        """ count number of trained parameters in a GaussianHMM model, assuming covariance_type=="diag"
        """
        n_components = len(hmm_model.startprob_)
        
        count = 0
        count += hmm_model.n_features * n_components # no. of mean parameters estimated
        count += hmm_model.n_features * n_components # no. of variance parameters estimated
        count += n_components * (n_components-1)     # no. of transition matrix parameters estimated
        count += n_components-1                      # no. of initial state probability parameters estimated

        return count

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # TODO implement model selection based on DIC scores
        best_dic=float("-inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            if self.verbose:
                print("Calculating DIC for {} with {} states".format(self.this_word, n))
            this_dic, this_model = self.calculate_dic(n)
            if this_dic > best_dic:
                best_dic = this_dic
                best_model = this_model
                best_n = n
        if self.verbose:
            if best_model is None:
                print("DIC couldn't create any model within given range of n_components")
            else:
                print("Best model for {} has {} states".format(self.this_word, best_n))
        return best_model
        
    def calculate_dic(self, n):
        """ Calculate DIC for a GaussianHMM with n components. Returns float("-inf") if exception
        was thrown.
        """
        # First calculate score for class in question
        try:
            hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            evidence = hmm_model.score(self.X, self.lengths)
        except:
            if self.verbose:
                print("    DIC calculation failure creating model on {} with {} states".format(self.this_word, n))
            return float("-inf"), None
        # Then calculate mean anti-evidence on other classes using that trained model
        mean_anti_evidence = (sum(hmm_model.score(X, lengths)
                                  for this_word, (X, lengths) in self.hwords.items()
                                  if this_word != self.this_word) /
                              (len(self.hwords.items()) - 1))
        dic = evidence - mean_anti_evidence
        if self.verbose:
            print("    DIC for model on {} with {} states is {}.".format(self.this_word, n, dic))
        return dic, hmm_model
    
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        """ Note we deal with a single sequence as a special case for this implementation of "cross-validation".
        Where we have a single sequence we can't do cross validation proper. But if we just return a selection
        failure in such cases we will not be able to use this class at all for my_recognizer (later in the project),
        since the dataset there has some words with only one sample. So, in this implementation, SelectorCV will fall
        back to in-sample maximum likelihood when asked to select a model topology for a word where there is only
        a single sample sequence available in the training data.
        """ 

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # TODO implement model selection using CV
        best_ll=float("-inf")
        n_splits = min(3, len(self.sequences)) # default 3 unless too few sequences

        # Take care of single-sequence training data -- see function docstring above for an explanation
        if n_splits==1:
            folds = [([0], [0])] # use the same sequence for both test and train
        else:
            folds = list(KFold(n_splits=n_splits, random_state=self.random_state).split(self.sequences))

        # Calculate out-of-sample log-likelihoods over folds
        ll_sums_over_folds = {n:0. for n in range(self.min_n_components, self.max_n_components+1)} # To keep track of log-likelihoods
        for fold_no, (cv_train_idx, cv_test_idx) in enumerate(folds):
            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
            for n in range(self.min_n_components, self.max_n_components+1):
                if ll_sums_over_folds[n]==float("-inf"):
                    # We've already given up on this n due to an error on a previous fold so skip
                    continue
                try:
                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    this_score = hmm_model.score(test_X, test_lengths)
                    if self.verbose:
                        print("model created for {} with {} states, fold #{}, l-l {}".format(self.this_word,
                                                                                             n,
                                                                                             fold_no,
                                                                                             this_score))
                except:
                    if self.verbose:
                        print("failure on {} with {} states, fold #{}".format(self.this_word, n, fold_no))
                    ll_sums_over_folds[n] = float("-inf") # Give up on this n if error on any fold
                    continue
                ll_sums_over_folds[n] += this_score
        if self.verbose:
            print("CV average log-likelihoods over all folds:")
            [print("n={}, ll={}".format(n, sum_ll/n_splits)) for n, sum_ll in ll_sums_over_folds.items()]
            
        # sort n's in descending order and try to train model on entire data, start with best n and if that
        # fails next best n and so on.
        n_descending_ll = sorted(ll_sums_over_folds, key=ll_sums_over_folds.get, reverse=True)
        for n in n_descending_ll:
            if ll_sums_over_folds[n]==float("-inf"):
                # then this n was no good in training
                break
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    this_score = hmm_model.score(self.X, self.lengths)
                    print("CV best model created for {} with {} states, log-likelihood {}".format(self.this_word,
                                                                                                  n,
                                                                                                  this_score))
                return hmm_model
            except:
                if self.verbose:
                    print("CV best model failure for {} with {} states".format(self.this_word, n))
                continue
        # if we get here then we were unable generate any model
        if self.verbose:
            print("CV unable to generate any valid model for {}".format(self.this_word))
        return None

            
