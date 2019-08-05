import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

from sklearn.manifold import TSNE
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataframe_utils
from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.dataset.Dataset
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    n_components = hyperparams.UniformInt(lower=1, upper=3, default = 2, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'dimension of the embedded space')  
    
    long_format = hyperparams.UniformBool(default = False, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether the input dataset is already formatted in long format or not")
    pass

class Tsne(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    '''
        Primitive that applies Hierarchical Density-Based Clustering or Density-Based Clustering 
        algorithms to time series data. This is an unsupervised, clustering primitive, but has been
        representend as a supervised classification problem to produce a compliant primitive. 

        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "15586787-80d5-423e-b232-b61f55a117ce",
        'version': __version__,
        'name': "tsne",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'Clustering', 'Dimensionality Reduction'],
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
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.clustering.hdbscan.Hdbscan',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})

        self.clf = TSNE(n_components = self.hyperparams['n_components'],random_state=self.random_seed)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length) containing new time series 

        Returns
        ----------
        Outputs
            The output is a transformed dataframe of X fit into an embedded space, n feature columns will equal n_components hyperparameter
        """ 
    
        hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
        metadata_inputs = ds2df_client.produce(inputs = inputs).value
        
        # temporary (until Uncharted adds conversion primitive to repo)
        if not self.hyperparams['long_format']:
            formatted_inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            formatted_inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)        

        # store information on target, index variable
        targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        target_names = [list(metadata_inputs)[t] for t in targets]
        index = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')

        # parse values from output of time series formatter
        n_ts = len(formatted_inputs.d3mIndex.unique())
        if n_ts == formatted_inputs.shape[0]:
            X_test = formatted_inputs.drop(columns = list(formatted_inputs)[index[0]])
            X_test = X_test.drop(columns = target_names).values
        else:
            ts_sz = int(formatted_inputs.shape[0] / n_ts)
            X_test = np.array(formatted_inputs.value).reshape(n_ts, ts_sz)

        # fit_transform data and create new dataframe
        n_components = self.hyperparams['n_components']

        col_names = ['dim_'+ str(c) for c in range(0,n_components)]
        tsne_df = d3m_DataFrame(pandas.DataFrame(self.clf.fit_transform(X_test), columns=[col_names]))
        tsne_df = pandas.concat([formatted_inputs[['d3mIndex','Class']], tsne_df], axis=1) #change this to point at index and target once this is running
        

        col_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        tsne_df.metadata = tsne_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        col_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        tsne_df.metadata = tsne_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        
        for c in range(2,n_components+2):
            col_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, c)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = ('dim_' + str(c-2))
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
            tsne_df.metadata = tsne_df.metadata.update((metadata_base.ALL_ELEMENTS, c), col_dict)

        df_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
        df_dict_1 = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
        df_dict['dimension'] = df_dict_1
        df_dict_1['name'] = 'columns'
        df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
        df_dict_1['length'] = (n_components+2)        
        tsne_df.metadata = tsne_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
        
        return CallResult(tsne_df)

if __name__ == '__main__':

    # Load data and preprocessing
    hyperparams_class = Tsne.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    tsne_client = Tsne(hyperparams=hyperparams_class.defaults().replace({'long_format':True}))
    filepath = 'file:///home/alexmably/datasets/seed_datasets_unsupervised/1491_one_hundred_plants_margin_clust/TEST/dataset_TEST/datasetDoc.json'
    print(filepath)
    test_dataset = container.Dataset.load(filepath)
    results = tsne_client.produce(inputs = test_dataset)
    print(results.value)
