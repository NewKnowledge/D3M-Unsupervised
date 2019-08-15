import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

from sklearn.cluster import SpectralClustering as SC
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataframe_utils, denormalize

from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 8, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description = 'The dimension of the projection space')

    n_init = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 10, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'Number of times the k-means algorithm will be run with different centroid seeds')

    n_neighbors = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 10, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'Number of neighbors when constructing the affintiy matrix using n-neighbors, ignored for affinity="rbf"')   

    affinity = hyperparams.Enumeration(default = 'rbf', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['rbf', 'nearest_neighbors'],
        description = 'method to construct affinity matrix')

    required_output = hyperparams.Enumeration(default = 'feature',semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['prediction','feature'],
        description = 'Determines whether the output is a dataframe with just predictions,\
            or an additional feature added to the input dataframe.')
    pass

class SpectralClustering(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    '''
        Primitive that applies the T-distributed stochastic neighbour embedding algorith to time series,
        unsupervised, supervised or semi-supervised datasets. 
        
        Training inputs: D3M dataset with features and labels, and D3M indices

        Outputs:For time series data, a dataframe of the inputs with t-SNE dimension columns appended 
                For everything else - D3M dataframe with t-SNE dimensions and D3M indices
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d13a4529-f0ba-44ee-a867-e0fdbb71d6e2",
        'version': __version__,
        'name': "tsne",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Clustering', 'Graph Clustering'],
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
        'python_path': 'd3m.primitives.clustering.spectral_clustering.SpectralClustering',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.SPECTRAL_CLUSTERING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.sc = SC(n_clusters=self.hyperparams['n_clusters'],n_init=self.hyperparams['n_init'],n_neighbors=self.hyperparams['n_neighbors'],affinity=self.hyperparams['affinity'],random_state=self.random_seed)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length) containing new time series 

        Returns
        ----------
        Outputs
            The output is a transformed dataframe of X fit into an embedded space, n feature columns will equal n_components hyperparameter
            For timeseries datasets the output is the dimensions concatenated to the timeseries filename dataframe
        """ 
    
        targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        target_names = [list(inputs)[t] for t in targets]
        index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        index_names = [list(inputs)[i] for i in index]
 
        X_test = inputs.drop(columns = list(inputs)[index[0]])
        X_test = X_test.drop(columns = target_names).values
        
        # special semi-supervised case - during training, only produce rows with labels
        series = inputs[target_names] != ''
        if series.any().any():
            inputs = dataframe_utils.select_rows(inputs, np.flatnonzero(series))
            X_test = X_test[np.flatnonzero(series)]

        sc_df = d3m_DataFrame(pandas.DataFrame(self.sc.fit_predict(X_test)))
        
        if self.hyperparams['required_output'] == 'feature':

            sc_df = d3m_DataFrame(pandas.DataFrame(self.sc.fit_predict(X_test), columns=['cluster_labels']))

            # just add last column of last column ('clusters')
            col_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = 'cluster_labels'
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
            sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
            df_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
            df_dict_1 = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
            df_dict['dimension'] = df_dict_1
            df_dict_1['name'] = 'columns'
            df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
            df_dict_1['length'] = 1        
            sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
                
            return CallResult(utils_cp.append_columns(inputs, sc_df))
        else:
            
            sc_df = d3m_DataFrame(pandas.DataFrame(self.sc.fit_predict(X_test), columns=[target_names[0]]))

            sc_df = pandas.concat([inputs.d3mIndex, sc_df], axis=1)

            col_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = index_names[0]
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
            sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
            
            col_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = str(target_names[0])
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
            
            df_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
            df_dict_1 = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
            df_dict['dimension'] = df_dict_1
            df_dict_1['name'] = 'columns'
            df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
            df_dict_1['length'] = 2        
            sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)

            return CallResult(sc_df)
            

if __name__ == '__main__':

    # Load data and preprocessing
    hyperparams_class = denormalize.DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    denorm = denormalize.DenormalizePrimitive(hyperparams = hyperparams_class.defaults())
    
    hyperparams_class = SpectralClustering.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    sc_client = SpectralClustering(hyperparams=hyperparams_class.defaults())
    filepath = 'file:///home/alexmably/datasets/seed_datasets_unsupervised/1491_one_hundred_plants_margin_clust/TEST/dataset_TEST/datasetDoc.json'
    test_dataset = container.Dataset.load(filepath)
    #read dataset into dataframe
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    test_dataframe = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)   
    #print(test_dataframe)
    
    #test_dataframetest_dataset = denorm.produce(inputs = test_dataframe).value
    results = sc_client.produce(inputs = test_dataframe)
    print(type(results.value))
    print(results.value)
