

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

"""## Datasets

"""

from sklearn.preprocessing import LabelEncoder


def get_encoded_dataset(df):
  LE = LabelEncoder()
  df[df.columns[-1]] = LE.fit_transform(df[df.columns[-1]])
  return df

def get_MaxAbsScaled_dataset (df):
  max_abs = preprocessing.MaxAbsScaler()
  scaled_df = max_abs.fit_transform(df.values)
  df.columns
  return pd.DataFrame(scaled_df,columns=df.columns)

def get_datasets(path: str):
  datasets = []
  datasets_in_path = os.listdir(path)
  for dataset in datasets_in_path:
    df = pd.read_csv(path + '/' + dataset, delimiter=',', header=None)
    # df = get_encoded_dataset(df)
    datasets.append(df)

  return datasets

TEST_SUFFIX = 'Test'
TRAIN_SUFFIX = 'Train'


def get_splited_datasets(path: str, name_list):
  datasets = []
  for name in name_list:
    df = pd.read_csv(path + '/' + name + TRAIN_SUFFIX + '.txt', delimiter=',', header=None)
    df = pd.concat([df ,pd.read_csv(path + '/' + name + TEST_SUFFIX + '.txt', delimiter=',', header=None)], ignore_index=True)
    datasets.append(df)
  return datasets

from sklearn.model_selection import train_test_split

def split_dataset(df):
  x = df.iloc[:,:-1]
  y = df.iloc[:,-1:]
  return train_test_split(x, y, test_size=0.20, shuffle=False)

from pandas.testing import assert_frame_equal

def verify_split(path: str, name_list):
  datasets_concat = get_splited_datasets(path, name_list)

  for name in name_list:
    df_train = pd.read_csv(path + '/' + name + TRAIN_SUFFIX + '.txt', delimiter=',', header=None)
    df_test = pd.read_csv(path + '/' + name + TEST_SUFFIX + '.txt', delimiter=',', header=None)

    x_train_original = df_train.iloc[:,:-1]
    y_train_original = df_train.iloc[:,-1:]
    x_test_original = df_test.iloc[:,:-1]
    y_test_original = df_test.iloc[:,-1:]

    x_train_concat, x_test_concat, y_train_concat, y_test_concat = split_dataset(get_splited_datasets(path, [name])[0])


    assert_frame_equal(x_train_original.reset_index(drop=True), x_train_concat.reset_index(drop=True))
    assert_frame_equal(y_train_original.reset_index(drop=True), y_train_concat.reset_index(drop=True))
    assert_frame_equal(x_test_original.reset_index(drop=True), x_test_concat.reset_index(drop=True))
    assert_frame_equal(y_test_original.reset_index(drop=True), y_test_concat.reset_index(drop=True))

"""## Scaler"""

def calculate_bounds(min=0, max=1, n = 3):
  return np.linspace(min, max, n + 1)

def calculate_column(column, n):
  bounds = calculate_bounds(n = n)
  bounds = bounds[1:]
  new_columns = [0 for _ in range(n)]
  old_value = column

  if (column < bounds[0]):
    new_columns[0] = column
    return new_columns
  else:
    new_columns[0] = bounds[0]


  for i in range(1, n):
    if column - bounds[i] >= 0:
      new_columns[i] = bounds[i] - bounds[i - 1]
    else:
      new_columns[i] = column - bounds[i - 1]
      return new_columns;
  return new_columns

def verify_calculation(data, column_name, new_column_names):
  count = data[column_name].eq(data[new_column_names].sum(axis=1)).eq(False).sum()
  if count > 0:
    raise BaseException('Calculation error: Can\'t scale dataset')

from pandas import DataFrame
def data_scaler(data: pd.DataFrame, n):
  df = DataFrame()
  y = data.iloc[:,-1:]
  data = data.drop(data.iloc[:,-1:],axis=1)
  for column_name, items in data.iteritems():
    new_columns = [(str(column_name) + '_' + str(i)) for i in range(n)]
    a = pd.DataFrame(data[column_name].apply(lambda x : calculate_column(x, n = n)).to_list(), columns=new_columns)
    df[new_columns] = pd.DataFrame(data[column_name].apply(lambda x : calculate_column(x, n = n)).to_list(), columns=new_columns)
    verify_calculation(data.join(df), column_name, new_columns)
  return df.join(y)

"""# Models"""

from enum import Enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class Models(Enum):
    DecisionTree = 1
    GaussianNB = 2
    AdaBoostClassifier = 3
    LogisticRegression = 4
    KNeighborsClassifier = 5
    RandomForest = 6
    SVMLinearKernel = 7
    SVMRBFKernel = 8

    @staticmethod
    def get_model(model):
      if model == Models.DecisionTree:
        return DecisionTreeClassifier()
      elif model == Models.GaussianNB:
        return GaussianNB()
      elif model == Models.AdaBoostClassifier:
        return AdaBoostClassifier()
      elif model == Models.LogisticRegression:
        return LogisticRegression()
      elif model == Models.KNeighborsClassifier:
        return KNeighborsClassifier()
      elif model == Models.RandomForest:
        return RandomForestClassifier()
      elif model == Models.SVMLinearKernel:
        return svm.SVC(kernel='linear')
      elif model == Models.SVMRBFKernel:
        return svm.SVC(kernel='rbf')

"""## Metrics"""

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
import time
import datetime
from google.colab import files

def init_metrics():
    return {
        'train': {
            'model': [],
            'precision_score': [],
            'recall_score': [],
            'f1_score': [],
            'accuracy_score': [],
            'cohen_kappa_score': [],
            'matthews_corrcoef': [],
        },
        'test': {
            'model': [],
            'precision_score': [],
            'recall_score': [],
            'f1_score': [],
            'accuracy_score': [],
            'cohen_kappa_score': [],
            'matthews_corrcoef': [],
        }
    }


def init_model(model: Models):
  return Models.get_model(model)

def max_ABS_scale(df):
  return get_MaxAbsScaled_dataset(df)

def calc_train_metrics(metrics, model, train_y, train_pred, is_multiclass):
  average = 'binary'
  if is_multiclass:
    average = 'micro'

  metrics['train']['model'].append(model)
  metrics['train']['precision_score'].append(precision_score(train_y, train_pred, average=average))
  metrics['train']['recall_score'].append(recall_score(train_y, train_pred, average=average))
  metrics['train']['f1_score'].append(f1_score(train_y, train_pred, average=average))
  metrics['train']['accuracy_score'].append(accuracy_score(train_y, train_pred))
  metrics['train']['cohen_kappa_score'].append(cohen_kappa_score(train_y, train_pred))
  metrics['train']['matthews_corrcoef'].append(matthews_corrcoef(train_y, train_pred))


def calc_test_metrics(metrics, model, test_y, test_pred, is_multiclass):
  average = 'binary'
  if is_multiclass:
    average = 'micro'

  metrics['test']['model'].append(model)
  metrics['test']['precision_score'].append(precision_score(test_y, test_pred, average=average))
  metrics['test']['recall_score'].append(recall_score(test_y, test_pred, average=average))
  metrics['test']['f1_score'].append(f1_score(test_y, test_pred, average=average))
  metrics['test']['accuracy_score'].append(accuracy_score(test_y, test_pred))
  metrics['test']['cohen_kappa_score'].append(cohen_kappa_score(test_y, test_pred))
  metrics['test']['matthews_corrcoef'].append(matthews_corrcoef(test_y, test_pred))

def run_experiment(dataset,
                   dataset_name: str=None, # if not null -> then call get_splited_datasets
                   dataset_path: str='/',
                   export_to_excel: bool=False,
                   scale_dataset: bool=False,
                   scale_num: int=3):
    """
  Function to run expiriments.

  Parameters
  ----------
  `dataset` : DataFrame
    Dataset that use for training
  `dataset_name` : str
    If not None then we will use get_splited_datasets() and don't use dataset
  `dataset_path` : str
    Specify the path when use `dataset_name`
  `export_to_excel`: bool
    If True we will create a file for metrics.
  `scale_dataset`: bool
    If True we will use custom scaling method.
  `scale_num`: int
    Only when `scale_dataset = True`. Count to split each column in x.
  """
    metrics = init_metrics()
    models = []


    if dataset_name != None:
      verify_split(dataset_path, [dataset_name])
      dataset = get_splited_datasets(dataset_path, [dataset_name])[0]


    # scale and encoding
    dataset = max_ABS_scale(dataset)
    LE = LabelEncoder()
    dataset[dataset.columns[-1]] = LE.fit_transform(dataset[dataset.columns[-1]])

    is_multiclass = len(LE.classes_) > 2


    # custom method
    if scale_dataset:
      dataset = data_scaler(dataset, scale_num)

    # split to test and train
    train_X, test_X, train_y, test_y = split_dataset(dataset)


    for model_type in (Models):
        model = init_model(model_type)


        # find and save model
        model.fit(train_X, train_y)
        models.append(model)

        # calculate metrics
        # test_pred = model.predict(test_X)
        # train_pred = model.predict(train_X)
        calc_train_metrics(metrics, model_type.name, train_y, model.predict(train_X), is_multiclass=is_multiclass)
        calc_test_metrics(metrics, model_type.name, test_y, model.predict(test_X), is_multiclass=is_multiclass)



    if export_to_excel:
        date_time_now = datetime.datetime.now()
        metrics_train_df = pd.DataFrame(data=metrics['train'])
        metrics_train_df.index.name = 'index'
        metrics_test_df = pd.DataFrame(data=metrics['test'])
        metrics_test_df.index.name = 'index'


        file_name = ''

        if dataset_name != None:
          file_name += dataset_name
        if scale_dataset:
          file_name += 'Scaled'
          file_name += 'Num_' + str(scale_num) + '_'
        else:
          file_name += 'DefaultScale'

        file_name += str(time.time()).replace('.', '-')

        with pd.ExcelWriter('experiment-run-{cascade}.xlsx'.format(cascade=str(file_name))) as writer:
            metrics_train_df.to_excel(writer, sheet_name='train')
            metrics_test_df.to_excel(writer, sheet_name='test')
        files.download('experiment-run-{cascade}.xlsx'.format(cascade=str(file_name)))

    return metrics, models

"""## Experiments"""

import os

binary_datasets_path = '/content/drive/MyDrive/BachFolder/datasets/binary'

multiclass_datasets_path = '/content/drive/MyDrive/BachFolder/datasets/multiclass'


BINARY_DATASETS_NAME_LIST = ['Blood', 'heart1', 'heart3']

MULTICLASS_DATASETS_NAME_LIST = ['breast', 'cmc', 'maternal']




# Datasets that will use to test

## Path
datasets_path = multiclass_datasets_path # OR multiclass_datasets_path

datasets_to_test = MULTICLASS_DATASETS_NAME_LIST # OR MULTICLASS_DATASETS_NAME_LIST
# OR use to test only 1 dataset:
# datasets_to_test = ['<name_of_dataset>']
# datasets_path = <path_to_dataset>

# Export to Excel
export_to_excl = False


# Custom scale params
is_scale = True
scale_num = 3


for i in range(2, 7):
  for dataset_name in datasets_to_test:
    run_experiment(None, dataset_name=dataset_name, dataset_path=datasets_path, export_to_excel=export_to_excl, scale_dataset=is_scale, scale_num=scale_num)
