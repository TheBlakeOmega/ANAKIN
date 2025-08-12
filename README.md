# ANAKIN

The repository contains code referred to the work:
(insert paper information)

## Code requirements
The code relies on the following python3.9+ libs.
Packages needed are:
* gensim==4.3.3
* hyperopt==0.2.7
* matplotlib~=3.9.0
* numpy~=1.26.4
* pandas~=2.2.2
* scikit_learn==1.5.2
* torch==2.5.0+cu124
* torch_geometric==2.6.1
* tqdm==4.66.5
* scikit-learn~=1.5.0
* networkx~=3.3
All required packages are displayed in _requirements.txt_ file.


## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run  ANAKIN

All configurations details must be detailed in _configuration.conf_ file:
```python
[DATASET]
originalTrainPathDataset=original/train
originalTestPathDataset=original/test
jsonTrainPathDataset=json_reduced/train
jsonTestPathDataset=json_reduced/test
graphTrainDatasetPath=graph/train
graphTestDatasetPath=graph/graph_test
graphTestDatasetToShowPath=graph_test_to_show


[SETTINGS]
top_n_tf_idf_strings=140
tfidf_embedding_size=100
path_models=results/models
pathHyperopt=results/hyperopt
path_gnn_explainer_results=results/GNNExplainer
trainEpochs=150
hyperopt_max_eval=2
earlyStoppingThresh=10
test_batch_size=64
LSTM_sequence_length=8
prediction_type=soft
#1 GCN, 2 GraphSage, 3 GAT, 4 graphconv
graphType=1
train_explainer_model=1
indexes_examples_to_isolate=9
TestConfidenceResultsFileName=04-02-2025_23-14-26_TestConfidenceResults
top_n_subgraph_to_take=5
file_names_examples_to_explain=_5_graph4.pkl,_5_graph3.pkl
preprocess_json_dataset_pipeline=0
train_word2vec_model_pipeline=0
create_graph_dataset_pipeline=0
train_gnn_model_pipeline=0
optimize_gnn_model_pipeline=0
test_gnn_model_pipeline=0
explain_gnn_model_pipeline=0
isolate_graphs_to_explain_pipeline=0
generate_graphs_to_explain_pipeline=1
```

_DATASET_ configuration sector includes all system paths where datasets are stored
_SETTINGS_ configuration sector includes all settings for run pipelines. Specifically, all configurations with suffix "__pipeline_" are boolean values that mean "1 = run pipeline" and "0 = skip pipeline" 

## Data
The dataset should be split into four folders: Benign of the test, Benign of the train, Malicious of the test and Malicious of the train.
The dataset must follow this step to train the model:
* preprocess the dataset running _preprocess_json_dataset_pipeline_
* train word2Vec model on train dataset running _train_word2vec_model_pipeline_
* generate graph dataset starting from preprocesssed json files running _create_graph_dataset_pipeline_

## Train model
After Dataset preprocessing, run _optimize_gnn_model_pipeline_ to train the model using hyperopt optimization.

## Test model
To test the model trained run _test_gnn_model_pipeline_ 

## Explain model
To generate an explanation of the model trained run:
* _explain_gnn_model_pipeline_ to generate a global explanation among the whole dataset
* _isolate_graphs_to_explain_pipeline_ to identify best examples to use as explanation, according results obtaind in the file specified by _TestConfidenceResultsFileName_ configuration
* _generate_graphs_to_explain_pipeline_ to generate files which includes graph objectss, instances of GraphWidget class from yfiles_jupyter_graphs library
* the notebook present in the repository, specifying, at line 22, the path of the directory in which the above pipeline stored the explanation graphs.
