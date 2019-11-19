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
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.Common'))
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
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.Common'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_4.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=3)
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: DISTIL/NK Storc primitive -> unsupervised clustering of records with a label (number of clusters set for kmeans algorithm)
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.k_means.Sloth'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_hyperparameter(name='nclusters', argument_type= ArgumentType.VALUE, data=100)
step_5.add_hyperparameter(name='n_init', argument_type= ArgumentType.VALUE, data=20)
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)
