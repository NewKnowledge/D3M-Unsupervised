import sys
import os.path
import numpy as np
import pandas as pd
import functools
import typing

from Sloth.cluster import KMeans
from tslearn.datasets import CachedDatasets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataframe_utils, denormalize

__author__ = 'Distil'
__version__ = '2.0.5'
__contact__ = 'mailto:nklabs@newknowledge.com'


Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

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
    n_init = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=10, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'Number of times the k-means algorithm \
        will be run with different centroid seeds. Final result will be the best output on n_init consecutive runs in terms of inertia')
    time_col_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing timestamps.'
    )
    value_col_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing the values associated with the timestamps.'
    )
    grouping_col_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing the values used to mark timeseries groups'
    )
    output_col_name = hyperparams.Hyperparameter[str](
        default='__cluster',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Name to assign to cluster column that is appended to the input dataset'
    )
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
        return

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[container.pandas.DataFrame]:
        """
        Parameters
        ----------
        inputs : D3M dataframe with associated metadata.

        Returns
        -------
        Outputs
            For unsupervised problems: The output is a dataframe containing a single column where each entry is the associated series' cluster number.
            For semi-supervised problems: The output is the input df containing an additional feature - cluster_label
        """
                # if the grouping col isn't set infer based on presence of grouping key
        grouping_key_cols = self.hyperparams.get('grouping_col_index', None)
        if grouping_key_cols is None:
            grouping_key_cols = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/GroupingKey',))
            if grouping_key_cols:
                grouping_key_col = grouping_key_cols[0]
            else:
                # if no grouping key is specified we can't split, and therefore we can't cluster.
                return None
        else:
            grouping_key_col = grouping_key_cols[0]

        # if the timestamp col isn't set infer based on presence of the Time role
        timestamp_cols = self.hyperparams.get('timestamp_col_index', None)
        if timestamp_cols is None:
            timestamp_col = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Time',))[0]
        else:
            timestamp_col = timestamp_cols[0]

        # if the value col isn't set, take the first integer/float attribute we come across that isn't the grouping or timestamp col
        value_cols = self.hyperparams.get('value_col_index', None)
        if value_cols is None:
            attribute_cols = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',))
            numerical_cols = inputs.metadata.list_columns_with_semantic_types(('http://schema.org/Integer', 'http://schema.org/Float'))

            for idx in numerical_cols:
                if idx != grouping_key_col and idx != timestamp_col and idx in attribute_cols:
                    value_col = idx
                    break
                value_col = -1
        else:
            value_col = value_cols[0]

        # split the long form series out into individual series and reshape for consumption
        # by ts learn
        groups = inputs.groupby(inputs.columns[grouping_key_col])
        values = [group.iloc[:, value_col].values for _, group in groups]
        keys = [group_name for group_name, _ in groups]

        # cluster the data
        self._kmeans = KMeans(self.hyperparams['nclusters'], self.hyperparams['algorithm'])
        self._kmeans.fit(values)
        clusters = self._kmeans.predict(values)
        
        # append the cluster column
        clusters = pd.DataFrame(list(zip(keys, self._kmeans.predict(values))), columns=('key', self.hyperparams['output_col_name']))
        outputs = inputs.join(clusters.set_index('key'), on=inputs.columns[grouping_key_col])
        outputs.metadata = outputs.metadata.generate(outputs)

        # update the new column metadata
        outputs.metadata = outputs.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        outputs.metadata = outputs.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')
        outputs.metadata = outputs.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'http://schema.org/Integer')

        return CallResult(outputs)