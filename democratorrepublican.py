import os
from nolearn.convnet import ConvNetFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import csv
from glob import glob
from collections import defaultdict
from sklearn import cross_validation
import warnings

from test.test_descrtut import defaultdict
warnings.filterwarnings("ignore", category=DeprecationWarning)

DECAF_IMAGENET_DIR = './imagenet/'
TRAIN_DATA_DIR = './data/'
TEST_DATA_DIR = './test/'

def get_dataset():
    r_dir = TRAIN_DATA_DIR + 'r/'
    r_filenames = [r_dir + fn for fn in os.listdir(r_dir)]
    d_dir = TRAIN_DATA_DIR + 'd/'
    d_filenames = [d_dir + fn for fn in os.listdir(d_dir)]

    labels = [0] * len(r_filenames) + [1] * len(d_filenames)
    filenames = r_filenames + d_filenames
    return shuffle(filenames, labels, random_state=0)


def get_testdataset():
    sorteddictionary = defaultdict(str)
    test_filenames = []

    images = glob(TEST_DATA_DIR + '*.png')
    for image in images:
        number = image.split('/')[-1]
        #print(image)
        number = int(number.split('.')[0])
        sorteddictionary[number] = image

    for k,v in sorteddictionary.items():
        test_filenames.append(v)

    return test_filenames

# help to install decafnet and nolearn
# http://pythonhosted.org/nolearn/convnet.html#example-dogs-vs-cats
def main():
    convnet = ConvNetFeatures(
    pretrained_params='imagenet.decafnet.epoch90',
    pretrained_meta='imagenet.decafnet.meta',
    )

    clf = linear_model.LogisticRegression(C=0.00001)
    
    pl = Pipeline([
        ('convnet', convnet),
        ('clf', clf),
        ])

    X, y = get_dataset()

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(X), k=5)

    results = []
    for traincv, testcv in cv:
        pl.fit(X[traincv], y[traincv])

        y_pred = pl.predict(X[testcv])
        print "Accuracy: %.3f" % accuracy_score(y[testcv], y_pred)
       
test = main()

