import sys
import os.path
import numpy as np
import pandas

from Sloth.cluster import KMeans
from sklearn.cluster import KMeans as sk_kmeans
from tslearn.datasets import CachedDatasets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataframe_utils, denormalize

from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '2.0.3'
__contact__ = 'mailto:nklabs@newknowledge.com'


Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'TimeSeriesKMeans', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['GlobalAlignmentKernelKMeans', 'TimeSeriesKMeans'],
        description = 'type of clustering algorithm to use')
    nclusters = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=3, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'number of clusters \
        to user in kernel kmeans algorithm')
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    n_init = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=10, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'Number of times the k-means algorithm \
        will be run with different centroid seeds. Final result will be the best output on n_init consecutive runs in terms of inertia')
    pass

class Storc(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Primitive that applies kmeans clustering to time series data. Algorithm options are 'GlobalAlignmentKernelKMeans'
        or 'TimeSeriesKMeans,' both of which are bootstrapped from the base library tslearn.clustering. This is an unsupervised, 
        clustering primitive, but has been represented as a supervised classification problem to produce a compliant primitive. 

        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "77bf4b92-2faa-3e38-bb7e-804131243a7f",
        'version': __version__,
        'name': "Sloth",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series','Clustering'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/D3M-Unsupervised",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [
            {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package': 'cython',
            'version': '0.29.7',
            },
            {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/D3M-Unsupervised.git@{git_commit}#egg=D3MUnsupervised'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
            }
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.clustering.k_means.Sloth',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.K_MEANS_CLUSTERING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._X_train = None          # training inputs
        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})
    
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Kmeans clustering algorithm using training data from set_training_data and hyperparameters
        '''
        self._kmeans.fit(self._X_train)
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data
        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length) containing training time series
        
        '''
        hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
        metadata_inputs = ds2df_client.produce(inputs = inputs).value
        if not self.hyperparams['long_format']:
            formatted_inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            formatted_inputs = ds2df_client.produce(inputs = inputs).value
        
        # store information on target, index variable
        targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        target_names = [list(metadata_inputs)[t] for t in targets]
        index = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        
        series = metadata_inputs[target_names] != ''
        self.clustering = 0 
        if not series.any().any():
            self.clustering = 1

        # load and reshape training data
        n_ts = len(formatted_inputs.d3mIndex.unique())
        if n_ts == formatted_inputs.shape[0]:
            self._kmeans = sk_kmeans(n_clusters = self.hyperparams['nclusters'], n_init = self.hyperparams['n_init'], random_state=self.random_seed)
            self._X_train_all_data = formatted_inputs.drop(columns = list(formatted_inputs)[index[0]])
            self._X_train = self._X_train_all_data.drop(columns = target_names).values
        else:
            self._kmeans = KMeans(self.hyperparams['nclusters'], self.hyperparams['algorithm'])
            ts_sz = int(formatted_inputs.shape[0] / n_ts)
            self._X_train = np.array(formatted_inputs.value).reshape(n_ts, ts_sz, 1)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[container.pandas.DataFrame]:
        """
        Parameters
        ----------
        inputs : Input pandas frame where each row is a series.  Series timestamps are store in the column names.

        Returns
        -------
        Outputs
            For unsupervised problems: The output is a dataframe containing a single column where each entry is the associated series' cluster number.
            For semi-supervised problems: The output is the input df containing an additional feature - cluster_label
        """
        hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
        metadata_inputs = ds2df_client.produce(inputs = inputs).value
        
        # temporary (until Uncharted adds conversion primitive to repo)
        if not self.hyperparams['long_format']:
            formatted_inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            formatted_inputs = ds2df_client.produce(inputs = inputs).value 
        
        # store information on target, index variable
        targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        target_names = [list(metadata_inputs)[t] for t in targets]
        index = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        index_names = [list(metadata_inputs)[i] for i in index]        

        # load and reshape training data
        n_ts = len(formatted_inputs.d3mIndex.unique())
        if n_ts == formatted_inputs.shape[0]:
            X_test = formatted_inputs.drop(columns = list(formatted_inputs)[index[0]])
            X_test = X_test.drop(columns = target_names).values
        else:
            ts_sz = int(formatted_inputs.shape[0] / n_ts)
            X_test = np.array(formatted_inputs.value).reshape(n_ts, ts_sz, 1)       
        
        # special semi-supervised case - during training, only produce rows with labels
        if self.clustering:
            
            sloth_df = d3m_DataFrame(pandas.DataFrame(self._kmeans.predict(X_test), columns = [target_names[0]]))

            sloth_df = pandas.concat([formatted_inputs.d3mIndex, sloth_df], axis=1)

            # first column ('d3mTndex')

            col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict['structural_type'] = type("1")
            col_dict['name'] = index_names[0]
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
            sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

            # second column ('Class')
            col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
            col_dict['structural_type'] = type("1")
            col_dict['name'] = target_names[0]
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
            
            df_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
            df_dict_1 = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
            df_dict['dimension'] = df_dict_1
            df_dict_1['name'] = 'columns'
            df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
            df_dict_1['length'] = 2         
            sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)

            return CallResult(sloth_df)

        else:
            series = metadata_inputs[target_names] != ''
            if series.any().any():
                metadata_inputs = dataframe_utils.select_rows(metadata_inputs, np.flatnonzero(series))
                X_test = X_test[np.flatnonzero(series)]
        
            sloth_df = d3m_DataFrame(pandas.DataFrame(self._kmeans.predict(X_test), columns=['cluster_labels']))

            # add clusters as a feature in the main dataframe - last column ('clusters')
            col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = 'cluster_labels'
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/CategoricalData')
            sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
            df_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
            df_dict_1 = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
            df_dict['dimension'] = df_dict_1
            df_dict_1['name'] = 'columns'
            df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
            df_dict_1['length'] = 1        
            sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
       
            return CallResult(utils_cp.append_columns(metadata_inputs, sloth_df))

if __name__ == '__main__':
    
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///home/alexmably/datasets/seed_datasets_unsupervised/1491_one_hundred_plants_margin_clust/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = denormalize.DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    denorm = denormalize.DenormalizePrimitive(hyperparams = hyperparams_class.defaults())
    input_dataset = denorm.produce(inputs = input_dataset).value

    hyperparams_class = Storc.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    storc_client = Storc(hyperparams = hyperparams_class.defaults().replace({'algorithm':'TimeSeriesKMeans','nclusters':100,'long_format':False, 'n_init':25}))
    storc_client.set_training_data(inputs = input_dataset, outputs = None)
    storc_client.fit()
    filepath = 'file:///home/alexmably/datasets/seed_datasets_unsupervised/1491_one_hundred_plants_margin_clust/TEST/dataset_TEST/datasetDoc.json'
    test_dataset = container.Dataset.load(filepath)
    test_dataset = denorm.produce(inputs = test_dataset).value
    results = storc_client.produce(inputs = test_dataset)
    print(results.value)
    
