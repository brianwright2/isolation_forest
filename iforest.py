
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def node_counter(t):
    """
    code taken from
    https://stackoverflow.com/questions/19187901/counting-number-of-nodes-in-a-binary-search-tree
    and altered for purposes of this project
    """
    if isinstance(t, IsolationTree):
        t = t.root
    count = 1
    if t is not None:
        if t.left is not None:
            count += node_counter(t.left)
        if t.right is not None:
            count += node_counter(t.right)
    return round(3/4*count)


def c(psi):
    if psi == 2:
        return 1
    elif psi > 2:
        return 2 * (np.log(psi - 1) + 0.5772156649) - 2 * (psi - 1) / psi
    else:
        return 0


def indiv_pl(row_n, t):
    path_length = 0
    if isinstance(t, IsolationTree):
        t = t.root
    while not isinstance(t, ExternalNode):
        path_length += 1
        if row_n[t.split_col] < t.split_val:
            t = t.left
        else:
            t = t.right
    if isinstance(t, ExternalNode):
        path_length += c(t.size)
        return path_length




class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = [IsolationTree(np.log2(sample_size)) for i in range(n_trees)]

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # randomly sample sample_size rows form X for each tree, and fit on that sample.
        for i in range(self.n_trees):
            X_sample = X[np.random.randint(X.shape[0], size=self.sample_size), :]
            self.trees[i].fit(X_sample)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = []
        # for each row in X
        for i in range(len(X)):
            # list to hold path lengths for a given row in all trees
            tree_length_list = []
            row = X[i, :]
            # for each tree in the forest
            for tree in self.trees:
                # get the path length for the given row and tree combination
                length = indiv_pl(row, tree)
                #if length is not None:
                tree_length_list.append(length)
            avg_length.append(np.mean(tree_length_list))
        return np.asarray(avg_length)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        pl = self.path_length(X)
        c_ = c(self.sample_size)
        return np.asarray([2 ** -(pl[i] / c_) for i in range(len(pl))])

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.asarray([1 if i >= threshold else 0 for i in scores])

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class IsolationTree:
    def __init__(self, max_height):
        self.depth = 0
        self.max_height = max_height
        self.root = None
        self.n_nodes = 0

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # grab row and cols
        n_rows, n_cols = X.shape
        # randomly select column
        choice = np.random.randint(n_cols)
        choice_col = X[:, choice]

        # randomly select split point in column
        random_split = np.random.uniform(min(choice_col), max(choice_col))

        # divide X into left and right by split value
        X_left = X[X[:, choice] < random_split]
        X_right = X[X[:, choice] >= random_split]

        # define the root of the tree, a decision node
        root = InternalNode(depth=self.depth + 1, mh=self.max_height,split_val=random_split,split_col=choice)
        root.split_val = random_split
        root.split_col = choice
        root.left = root.add_new_child(X_left)
        root.right = root.add_new_child(X_right)
        self.root = root

        # after the tree is fit, node_counter walks the tree and counts the nodes
        self.n_nodes = node_counter(self)

        return self.root


class ExternalNode:
    def __init__(self, val, depth):
        self.left = None
        self.right = None
        if type(val) == int:
            self.size = 1
        else:
            self.size = len(val)
        self.depth = depth
        self.score = 0


class InternalNode:
    def __init__(self, depth, mh,split_val=None,split_col=None):
        self.mh = int(mh)
        self.left = None
        self.right = None
        self.depth = int(depth)
        self.split_val = split_val
        self.split_col = split_col

    def add_new_child(self, X):
        # check if child should be leaf node
        if X.shape[0] <= 1 or self.depth >= self.mh:
            node = ExternalNode(val=X, depth=self.depth + 1)
        # if not, re iterate the decision node process
        else:
            n_rows, n_cols = X.shape
            choice = np.random.randint(n_cols)
            choice_col = X[:, choice]
            random_split = np.random.uniform(min(choice_col), max(choice_col))
            X_left = X[X[:, choice] < random_split]
            X_right = X[X[:, choice] >= random_split]
            node = InternalNode(depth=self.depth + 1, mh=self.mh,split_val=random_split,split_col=choice)
            node.left = node.add_new_child(X_left)
            node.right = node.add_new_child(X_right)
        return node

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    ...
    threshold_range = [i/200 for i in range(200,1,-1)]
    for t in threshold_range:
        y_pred = np.asarray([1 if i >= t else 0 for i in scores])
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return t, FPR



