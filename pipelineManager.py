import os
import pickle
import dill
import torch
import traceback
from datetime import datetime
import numpy as np
import utils
from GNNExplainer import GraphNetworkExplainer
from jsonDataset import JsonDataset
from graphDataset import GraphDataset
from embeddingModel import EmbeddingModel
from graphConvolutionalNetworkClassifier import GraphNetwork
from plot import computeTestResults
from utils import write_to_result_file, generate_examples_confidences_csv, generate_example_prediction_info_xlsx
from hyperopt import tpe, hp, Trials, fmin
import random
import pandas as pd

seed = 0
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class PipeLineManager:

    def __init__(self, configuration, dataset_configuration):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def runPipeline(self):
        """
        This method runs the pipeline chosen according to the configuration file
        """
        if self.configuration['preprocess_json_dataset_pipeline'] == '1':
            try:
                print("START preprocess_json_dataset_pipeline EXECUTION")
                write_to_result_file("START preprocess_json_dataset_pipeline EXECUTION")
                self._preProcessJSONDataset()
                print("JSON dataset preprocessed")
                write_to_result_file("JSON dataset preprocessed")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during JSON dataset preprocessing")
        if self.configuration['train_word2vec_model_pipeline'] == '1':
            try:
                print("START train_word2vec_model_pipeline EXECUTION")
                write_to_result_file("START train_word2vec_model_pipeline EXECUTION")
                self._trainWord2VecModel()
                print("Word2Vec generated")
                write_to_result_file("Word2Vec generated")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Word2Vec model training")
        if self.configuration['create_graph_dataset_pipeline'] == '1':
            try:
                print("START create_graph_dataset_pipeline EXECUTION")
                write_to_result_file("START create_graph_dataset_pipeline EXECUTION")
                self._generateGraphDataset()
                print("Graph dataset generated")
                write_to_result_file("Graph dataset generated")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Graph dataset creation")
        if self.configuration['train_gnn_model_pipeline'] == '1':
            try:
                print("START train_gnn_model_pipeline EXECUTION")
                write_to_result_file("START train_gnn_model_pipeline EXECUTION")
                self._trainGNNModel()
                print("GNN model trained")
                write_to_result_file("GNN model trained")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN training")
        if self.configuration['optimize_gnn_model_pipeline'] == '1':
            try:
                print("START optimize_gnn_model_pipeline EXECUTION")
                write_to_result_file("START optimize_gnn_model_pipeline EXECUTION")
                self._optimizeGNNModel()
                print("GNN model optimized")
                write_to_result_file("GNN model optimized")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN optimization")
        if self.configuration['test_gnn_model_pipeline'] == '1':
            try:
                print("START test_gnn_model_pipeline EXECUTION")
                write_to_result_file("START test_gnn_model_pipeline EXECUTION")
                self._testGNNModel()
                print("GNN model tested")
                write_to_result_file("GNN model tested")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN testing")
        if self.configuration['explain_gnn_model_pipeline'] == '1':
            try:
                print("START explain_gnn_model_pipeline EXECUTION")
                write_to_result_file("START explain_gnn_model_pipeline EXECUTION")
                self._explainGCNModel()
                print("GNN model explained")
                write_to_result_file("GNN model explained")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN testing")
        if self.configuration['isolate_graphs_to_explain_pipeline'] == '1':
            try:
                print("START isolate_graphs_to_explain_pipeline EXECUTION")
                write_to_result_file("START isolate_graphs_to_explain_pipeline EXECUTION")
                self._selectGraphExamplesToExplainFromTestSet()
                print("Graphs isolated")
                write_to_result_file("Graphs isolated")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Graphs isolation")
        if self.configuration['generate_graphs_to_explain_pipeline'] == '1':
            try:
                print("START generate_graphs_to_explain_pipeline EXECUTION")
                write_to_result_file("START generate_graphs_to_explain_pipeline EXECUTION")
                self._generateWidgetGraphFilesToShow()
                print("Graphs widget created")
                write_to_result_file("Graphs widget created")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Graphs widget creation")

    def _preProcessJSONDataset(self):
        """
        This method runs the pipeline topre process the JSON dataset
        It filters the dataset by allowed packages and TF-IDF top strings
        """
        allowed_packages = [
            "Landroid/accounts",
            "Landroid/app",
            "Landroid/bluetooth",
            "Landroid/content",
            "Landroid/location",
            "Landroid/media",
            "Landroid/net",
            "Landroid/nfc",
            "Landroid/provider",
            "Landroid/telecom",
            "Landroid/telephony",
        ]

        train_dataset = JsonDataset(self.ds_configuration['originalTrainPathDataset'])
        print("Filtering train dataset by allowed packages")
        write_to_result_file("Filtering train dataset by allowed packages")
        train_dataset.filterDatasetByPackages(allowed_packages, self.ds_configuration['jsonTrainPathDataset'])
        train_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        print("Computing TF-IDF")
        write_to_result_file("Computing TF-IDF")
        if not os.path.exists("results/" + "top_" + self.configuration['top_n_tf_idf_strings']
                              + "_tdf_idf_strings.pkl"):
            top_tfidf_string = train_dataset.compute_tf_idf(int(self.configuration['top_n_tf_idf_strings']), "results/"
                                                            + "top_" + self.configuration['top_n_tf_idf_strings']
                                                            + "_tdf_idf_strings.pkl")
        else:
            with (open("results/" + "top_" + self.configuration['top_n_tf_idf_strings'] + "_tdf_idf_strings.pkl", 'rb')
                  as file):
                print("Loading nodes top " + str(self.configuration['top_n_tf_idf_strings']) + "tf-idf strings")
                top_tfidf_string = pickle.load(file)
                file.close()
        print("Filtering train dataset by TFIDF top strings")
        write_to_result_file("Filtering train dataset by TFIDF top strings")
        train_dataset.filterDatasetByTFIDF(top_tfidf_string, self.ds_configuration['jsonTrainPathDataset'])
        print("Deleting files with empty properties from train dataset")
        write_to_result_file("Deleting files with empty properties from train dataset")
        train_dataset.delete_files_with_empty_properties()

        test_dataset = JsonDataset(self.ds_configuration['originalTestPathDataset'])
        print("Filtering test dataset by allowed packages")
        write_to_result_file("Filtering test dataset by allowed packages")
        test_dataset.filterDatasetByPackages(allowed_packages, self.ds_configuration['jsonTestPathDataset'])
        test_dataset = JsonDataset(self.ds_configuration['jsonTestPathDataset'])
        print("Filtering test dataset by TFIDF top strings")
        write_to_result_file("Filtering test dataset by TFIDF top strings")
        test_dataset.filterDatasetByTFIDF(top_tfidf_string, self.ds_configuration['jsonTestPathDataset'])

    def _trainWord2VecModel(self):
        """
        This method runs the pipeline to train the Word2Vec model
        It extracts the sequences from the JSON dataset and trains the model
        """
        print("Extracting sequences from dataset in " + self.ds_configuration['jsonTrainPathDataset'])
        write_to_result_file("Extracting sequences from dataset in " + self.ds_configuration['jsonTrainPathDataset'])
        train_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        extracted_sequences = train_dataset.extract_sequences()
        print("Training Word2Vec model")
        write_to_result_file("Training Word2Vec model")
        embedding_model = EmbeddingModel()
        embedding_model.train(extracted_sequences, int(self.configuration['tfidf_embedding_size']))
        print("Saving Word2Vec model")
        write_to_result_file("Saving Word2Vec model")
        embedding_model.saveModel(self.configuration['path_models'])
        print(embedding_model.stringEmbeddingModel.wv)
        write_to_result_file(str(embedding_model.stringEmbeddingModel.wv))

    def _generateGraphDataset(self):
        """
        This method runs the pipeline to generate the graph dataset
        It loads the Word2Vec model and generates the graph dataset from the JSON dataset
        """
        print("Loading Word2Vec model")
        write_to_result_file("Loading Word2Vec model")
        embedding_model = EmbeddingModel()
        embedding_model.loadModel(self.configuration['path_models'])

        print("Generating train graph dataset")
        write_to_result_file("Generating train graph dataset")
        train_json_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        train_graph_dataset = GraphDataset(self.ds_configuration['graphTrainDatasetPath'])
        train_graph_dataset.delete()
        train_graph_dataset.generateGraphDataset(train_json_dataset, embedding_model)

        print("Generating test graph dataset")
        write_to_result_file("Generating test graph dataset")
        test_json_dataset = JsonDataset(self.ds_configuration['jsonTestPathDataset'])
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])
        test_graph_dataset.delete()
        test_graph_dataset.generateGraphDataset(test_json_dataset, embedding_model)

    def _trainGNNModel(self):
        """
        This method runs the pipeline to train the GNN model
        It loads the graph dataset and trains the model according to the parameters in the configuration file
        """
        print("Creating train and validation dataloaders")
        write_to_result_file("Creating train and validation dataloaders")
        train_graph_dataset = GraphDataset(self.ds_configuration['graphTrainDatasetPath'])
        train_data_loader, validation_data_loader = train_graph_dataset.createTrainAndValidationDataLoader(
            train_batch_size=512, validation_batch_size=512)

        model = GraphNetwork()
        save_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        print("Starting GCN training on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        write_to_result_file("Starting GCN training on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        start_train_time = np.datetime64(datetime.now())
        space = {
            'hidden_dim': 256,
            'dropout': 0.4,
            'n_layers': 1,
            'n_convs': 1,
            'dataset': train_graph_dataset,
            'device': self.device,
            'save_model_path': save_path,
            'learning_rate': 0.01,
            'batch_size': 512,
            'epochs': int(self.configuration['trainEpochs']),
            'earlyStoppingThresh': int(self.configuration['earlyStoppingThresh']),
            'input_size': int(self.configuration['tfidf_embedding_size']),
        }
        scores = model.train(train_data_loader, validation_data_loader, space)
        end_train_time = np.datetime64(datetime.now())
        write_to_result_file("Trained model result:" + "\nLoss: " + str(scores['train_loss']) +
                             "\nAccuracy: " + str(scores['train_accuracy']) +
                             "\nVal Loss: " + str(scores['validation_loss']) +
                             "\nVal Accuracy: " + str(scores['validation_accuracy']) +
                             "\nTrain time: " + str(utils.convertTimeDelta(start_train_time, end_train_time)))

    def _testGNNModel(self):
        """
        This method runs the pipeline to test the GNN model
        """
        print("Loading GNN model")
        write_to_result_file("Loading GNN model")
        model = GraphNetwork()
        load_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        model.loadModel(load_path, self.device)
        model.setModelMode('eval')
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])

        print("Test phase started")
        write_to_result_file("Test phase started")
        start_test_time = np.datetime64(datetime.now())
        real_labels = []
        predicted_labels = []
        example_confidences = []
        example_index = 1
        while True:
            testDataLoader, _ = test_graph_dataset.createTestDataloader(start_sub_string="_" + str(example_index) + "_",
                                                                        batch_size=int(
                                                                            self.configuration['test_batch_size']))
            if testDataLoader is not None:
                with torch.no_grad():
                    predicted_label, real_label, confidence = model.predict_example(testDataLoader, self.device,
                                                                                    prediction_type=self.configuration[
                                                                                        'prediction_type'])
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                example_confidences.append(confidence)
            else:
                break
            example_index += 1
            if example_index % 10 == 0:
                print(str(example_index) + ' test examples computed')
                write_to_result_file(str(example_index) + ' test examples computed')
        end_test_time = np.datetime64(datetime.now())

        computeTestResults(real_labels, predicted_labels, self.configuration['prediction_type'],
                           end_test_time - start_test_time)
        generate_examples_confidences_csv(predicted_labels, real_labels, example_confidences,
                                          "results/" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") +
                                          "_TestConfidenceResults.xlsx")

    def _optimizeGNNModel(self):
        """
        This method runs the pipeline to optimize train's parameters
        """
        model = GraphNetwork()
        save_result_path = (self.configuration['pathHyperopt'] + "/" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                            + "_ConvGCN_hyperopt_result.csv")
        save_model_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        print("Creating train and validation dataloaders")
        write_to_result_file("Creating train and validation dataloaders")
        train_graph_dataset = GraphDataset(self.ds_configuration['graphTrainDatasetPath'])

        space = {
            'hidden_dim': hp.choice('hidden_dim', [32, 64, 128, 256]),
            'dropout': hp.uniform("dropout", 0.0, 0.5),
            'n_layers': hp.choice('n_layers', [1, 2, 3]),
            'n_convs': hp.choice('n_convs', [1, 2, 3]),
            'dataset': train_graph_dataset,
            'device': self.device,
            'save_results_path': save_result_path,
            'save_model_path': save_model_path,
            'learning_rate': hp.uniform("learning_rate", 0.0001, 0.01),
            'batch_size': hp.choice("batch_size", [32, 64, 128, 256, 512]),
            'epochs': int(self.configuration['trainEpochs']),
            'earlyStoppingThresh': int(self.configuration['earlyStoppingThresh']),
            'input_size': int(self.configuration['tfidf_embedding_size']),
        }

        print("Starting GCN optimization on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        write_to_result_file("Starting GCN optimization on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        trials = Trials()
        fmin(model.optimizeParameters, space, trials=trials, algo=tpe.suggest,
             max_evals=int(self.configuration['hyperopt_max_eval']))

    def _explainGCNModel(self):
        """
        This method runs the pipeline to explain the GNN model
        It loads the GNN model, trains the GNN explainer model and generates the explanations for the test dataset
        """
        print("Loading GNN model")
        write_to_result_file("Loading GNN model")
        model = GraphNetwork()
        load_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        model.loadModel(load_path, self.device)
        model.setModelMode("eval")
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])
        xlsx_save_path = (self.configuration['path_gnn_explainer_results'] + "/" +
                          datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + "_GNNExplainerResults.xlsx")
        explainer = GraphNetworkExplainer(model, "results/" + "top_" + self.configuration['top_n_tf_idf_strings']
                                          + "_tdf_idf_strings.pkl")
        if self.configuration['train_explainer_model'] == '1':
            explainer.train_explainer(self.configuration['path_models'] + "/GNNExplainerModel.pkl")
        else:
            explainer.load_explainer(self.configuration['path_models'] + "/GNNExplainerModel.pkl")
        explainer.explain_graphs(test_graph_dataset, self.device, xlsx_save_path,
                                 self.configuration['path_gnn_explainer_results'] + "/" +
                                 "_Nodes_importance_scores.pkl",
                                 self.configuration['path_gnn_explainer_results'] + "/" +
                                 "_edges_importance_scores.pkl")

    def _selectGraphExamplesToExplainFromTestSet(self):
        """
        This method runs the pipeline to select the examples to explain from the test set
        It loads the GNN model and the test dataset and selects the examples to explain according to the parameters in the configuration file
        """
        print("Loading GNN model")
        write_to_result_file("Loading GNN model")
        model = GraphNetwork()
        load_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        model.loadModel(load_path, self.device)
        model.setModelMode('eval')

        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])

        print("Load example indexes from " + self.configuration['TestConfidenceResultsFileName'])
        well_predicted_examples = pd.read_excel(
            pd.ExcelFile("results/" + self.configuration['TestConfidenceResultsFileName'] + ".xlsx"),
            sheet_name="Test results")
        well_predicted_examples = well_predicted_examples[(well_predicted_examples['Predicted_labels'] == 1) &
                                                          (well_predicted_examples['Confidence'] != 1.0)]

        examples_info = []
        example_computed = 0
        print("examples to compute: " + str(len(well_predicted_examples["Example_index"].tolist())))
        for example_index in well_predicted_examples["Example_index"].tolist():
            testDataLoader, filtered_indexes = test_graph_dataset.createTestDataloader(start_sub_string="_" +
                                                                                                        str(example_index) + "_",
                                                                                       batch_size=int(
                                                                                           self.configuration[
                                                                                               'test_batch_size']))
            if testDataLoader is not None:
                info = model.predict_example_with_top_n_subgraph_infos(
                    testDataLoader, test_graph_dataset, int(self.configuration['top_n_subgraph_to_take']),
                    example_index, filtered_indexes, self.device)
                examples_info.append(info)
            example_computed += 1
            if example_computed % 10 == 0:
                print(str(example_computed) + ' test examples computed')
                write_to_result_file(str(example_computed) + ' test examples computed')
        generate_example_prediction_info_xlsx(examples_info, int(self.configuration['top_n_subgraph_to_take']),
                                              "results/" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") +
                                              "_MalwareTop" + self.configuration['top_n_subgraph_to_take']
                                              + "SubGraphsInfo.xlsx")

    def _generateWidgetGraphFilesToShow(self):
        """
        This method runs the pipeline to generate the widget graph files to show
        It loads the GNN model and the test dataset and generates the widget graph files to show according to the parameters in the configuration file
        """
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])

        file_names_example_to_explain = [file_name for file_name in
                                         self.configuration['file_names_examples_to_explain'].split(",")]
        print("Number of examples to isolate: " + str(len(file_names_example_to_explain)))
        print("Examples to isolate: " + str(file_names_example_to_explain))

        print("Loading GNN model")
        model = GraphNetwork()
        load_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        model.loadModel(load_path, self.device)
        print("Loading GNN Explainer model")
        explainer = GraphNetworkExplainer(model, "results/" + "top_" + self.configuration['top_n_tf_idf_strings']
                                          + "_tdf_idf_strings.pkl")
        explainer.load_explainer(self.configuration['path_models'] + "/GNNExplainerModel.pkl",
                                 self.configuration['path_gnn_explainer_results'] + "/" +
                                 "_Nodes_importance_scores.pkl",
                                 self.configuration['path_gnn_explainer_results'] + "/" +
                                 "_edges_importance_scores.pkl")
        print("Saving graph widget to display")
        if not os.path.exists(self.ds_configuration['graphTestDatasetToShowPath']):
            os.makedirs(self.ds_configuration['graphTestDatasetToShowPath'])
        for graph_file_name in file_names_example_to_explain:
            graph = test_graph_dataset.get_by_filename(graph_file_name)

            if graph is not None:
                graph_to_display = explainer.build_visualization_graph(graph, self.device)
                with open(self.ds_configuration['graphTestDatasetToShowPath'] + "/" + graph_file_name, 'wb') as file:
                    dill.dump(graph_to_display, file)
