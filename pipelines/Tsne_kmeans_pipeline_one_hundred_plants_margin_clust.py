from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1 column parser -> labeled semantic types to data types
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
pipeline_description.add_step(step_1)

#Step 2 mapped to operate on a dataset object instead of a dataframe object
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_2.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=1)
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3 imputer -> imputes null values based on mean of column
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
pipeline_description.add_step(step_3)

# Step 4 mapped to operate on a dataset object instead of a dataframe object
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_4.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=3)
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5 DISTIL/NK t-SNE primitive -> dimensionality reduction
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_hyperparameter(name='n_components', argument_type=ArgumentType.VALUE,data=3)
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: DISTIL primitive -> unsupervised clustering, n_clusters set for kmeans, cluster_col_name is the target name
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.k_means.DistilKMeans'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_hyperparameter(name='n_clusters', argument_type= ArgumentType.VALUE, data=100)
step_6.add_hyperparameter(name = 'cluster_col_name', argument_type=ArgumentType.VALUE, data='Class')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)
