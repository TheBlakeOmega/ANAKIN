import heapq

from utils import write_to_result_file
from sklearn.metrics import accuracy_score
from torch.nn import Module, BCELoss
from torch_geometric.nn import GCNConv, Linear, global_mean_pool, SAGEConv, GATConv, GraphConv, GATv2Conv, CGConv, \
    GCN2Conv
from torch.nn.functional import sigmoid, relu, dropout
import torch
import pickle
import numpy as np
from datetime import datetime
from hyperopt import STATUS_OK
from torch import nn
import csv

SavedParameters = []
best_loss = np.inf
best_model = None


class GraphNetwork:
    """
    A class to manage the training, evaluation, and optimization of a graph neural network.
    """

    def __init__(self):
        """
        Initializes the GraphNetwork object.
        """
        self.model = None

    def setModelMode(self, mode):
        """
        Sets the mode of the model (train or eval).

        Args:
            mode (str): Mode to set ('train' or 'eval').

        Returns:
            None
        """
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()

    def setModelLocation(self, device):
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): Device to move the model to.

        Returns:
            None
        """
        self.model.to(device)

    def train(self, train_data_loader, validation_data_loader, params, save_path=None):
        """
        Trains the graph neural network.

        Args:
            train_data_loader (DataLoader): DataLoader for training data.
            validation_data_loader (DataLoader): DataLoader for validation data.
            params (dict): Dictionary of training parameters.
            save_path (str, optional): Path to save the trained model. Defaults to None.

        Returns:
            dict: Dictionary containing training and validation metrics.
        """
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        early_stopping_thresh = params['earlyStoppingThresh']
        device = params['device']
        self.model = GraphNeuralNetwork(params['input_size'], params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=300)

        criterion = BCELoss()
        best_val_loss = np.inf
        best_val_acc = np.inf
        best_train_loss = 0
        best_train_acc = 0
        worst_loss_times = 0

        for epoch in range(epochs):
            print('Training epoch ' + str(epoch) + ':')
            # Train on batches
            total_acc = 0
            val_loss = 0
            val_acc = 0
            total_loss = 0
            self.model.train()

            for batch in train_data_loader:
                batch.to(device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                out = out.squeeze()
                true_labels = batch.y.to(device)
                loss = criterion(out, true_labels)
                total_loss += float(loss)
                total_acc += accuracy_score(true_labels.tolist(), (out >= 0.5).tolist())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                print('Validating epoch ' + str(epoch) + ':')
                self.model.eval()
                for val_batch in validation_data_loader:
                    val_batch.to(device)
                    out = self.model(val_batch.x, val_batch.edge_index, val_batch.edge_attr, val_batch.batch)
                    out = out.squeeze()
                    true_labels = val_batch.y.to(device)
                    loss = criterion(out, true_labels)
                    val_loss += float(loss)
                    val_acc += accuracy_score(true_labels.tolist(), (out >= 0.5).tolist())

            total_loss /= len(train_data_loader)
            total_acc /= len(train_data_loader)
            val_loss /= len(validation_data_loader)
            val_acc /= len(validation_data_loader)

            print(f'TRM: Epoch {epoch + 1:>3} '
                  f'| Train Loss: {total_loss:.3f} '
                  f'| Train Acc: {total_acc:.3f} '
                  f'| Val Loss: {val_loss:.3f} '
                  f'| Val Acc: {val_acc:.3f}')
            write_to_result_file(f'TRM: Epoch {epoch + 1:>3} '
                                 f'| Train Loss: {total_loss:.3f} '
                                 f'| Train Acc: {total_acc:.3f} '
                                 f'| Val Loss: {val_loss:.3f} '
                                 f'| Val Acc: {val_acc:.3f}')

            scheduler.step()

            if round(val_loss, 4) >= best_val_loss:
                worst_loss_times += 1
            else:
                worst_loss_times = 0
                best_val_loss = round(val_loss, 4)
                best_val_acc = round(val_acc, 4)
                best_train_loss = round(total_loss, 4)
                best_train_acc = round(total_acc, 4)
                # save best model's weights
                torch.save(self.model.state_dict(), 'tmp/temp_best_model_state_dict.pt')

            if worst_loss_times == early_stopping_thresh:
                break

        # reload best model's weights
        self.model.load_state_dict(torch.load('tmp/temp_best_model_state_dict.pt', map_location=torch.device(device)))

        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("Saving model in pickle object")
                write_to_result_file("Saving model in pickle object")
                pickle.dump(self.model, file)
                print("Model saved")
                file.close()

        scores = {
            'train_loss': best_train_loss,
            'train_accuracy': best_train_acc,
            'validation_loss': best_val_loss,
            'validation_accuracy': best_val_acc,
            'epochs': epoch + 1
        }

        return scores

    def predict_example(self, test_dataloader, device, prediction_type='soft'):
        """
        Predicts the label for a single example using the graph neural network.

        Args:
            test_dataloader (DataLoader): DataLoader containing the test example.
            device (torch.device): Device to perform the prediction on.
            prediction_type (str, optional): Type of prediction ('soft' or 'hard'). Defaults to 'soft'.

        Returns:
            tuple: Predicted label, real label, and confidence score.
        """
        predictions_on_sub_graphs = []
        with torch.no_grad():
            for batch in test_dataloader:
                if hasattr(batch, 'x') and batch.x is not None:
                    batch.to(device)
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    out.squeeze(dim=0)
                    predictions_on_sub_graphs.extend(out.tolist())
                else:
                    predictions_on_sub_graphs.append(1)
        real_label = batch.y.int().tolist()[0] if hasattr(batch, 'x') else batch.y.int().item()
        if prediction_type == 'soft':
            return 1 if np.mean(predictions_on_sub_graphs) > 0.5 else 0, real_label, np.mean(predictions_on_sub_graphs)
        if prediction_type == 'hard':
            if len(predictions_on_sub_graphs) > 1:
                predictions_on_sub_graphs = [1 if prediction[0] > 0.5 else 0 for prediction in
                                             predictions_on_sub_graphs]
            elif len(predictions_on_sub_graphs) == 1 and predictions_on_sub_graphs[0] != 1:
                prediction = predictions_on_sub_graphs[0]
                predictions_on_sub_graphs = [1 if prediction[0] > 0.5 else 0]
            else:
                print('no graphs')
                predictions_on_sub_graphs = [1]
            return max(predictions_on_sub_graphs, key=predictions_on_sub_graphs.count), real_label, np.mean(predictions_on_sub_graphs)

    def predict_example_with_top_n_subgraph_infos(self, test_dataloader, graph_dataset, top_n, example_index,
                                                  subgraphs_indexes, device):
        """
        Predicts the label for a single example and retrieves information about the top N subgraphs.

        Args:
            test_dataloader (DataLoader): DataLoader containing the test example.
            graph_dataset (GraphDataset): Dataset containing the graph data.
            top_n (int): Number of top subgraphs to retrieve information for.
            example_index (int): Index of the example being predicted.
            subgraphs_indexes (list): List of subgraph indices.
            device (torch.device): Device to perform the prediction on.

        Returns:
            example_info: Information about the example and its top N subgraphs.
        """
        predictions_on_sub_graphs = []
        with torch.no_grad():
            for batch in test_dataloader:
                if hasattr(batch, 'x') and batch.x is not None:
                    batch.to(device)
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    out.squeeze(dim=0)
                    predictions_on_sub_graphs.extend([x[0] for x in out.tolist()])
                else:
                    predictions_on_sub_graphs.append(1)
        top_n_with_indices = sorted(heapq.nlargest(top_n, list(zip(subgraphs_indexes, predictions_on_sub_graphs)),
                                                   key=lambda x: x[1]),
                                    key=lambda x: x[1], reverse=True)
        example_info = [example_index]
        for subgraph_index, prediction_value in top_n_with_indices:
            graph = graph_dataset.get(subgraph_index)
            incoming_nodes = graph.edge_index[1]
            example_info.extend([
                graph_dataset.get_filename_by_index(subgraph_index),
                graph.x.size(0),
                graph.edge_index.size(1),
                (len(incoming_nodes) != len(set(incoming_nodes.tolist()))),
                prediction_value
            ])
        return example_info

    def optimizeParameters(self, params):
        """
        Optimizes the hyperparameters of the graph neural network.

        Args:
            params (dict): Dictionary containing the parameters for optimization, including:
                - dataset (GraphDataset): Dataset object for training and validation.
                - batch_size (int): Batch size for training and validation.
                - learning_rate (float): Learning rate for the optimizer.
                - hidden_dim (int): Dimension of the hidden layers.
                - dropout (float): Dropout rate.
                - n_layers (int): Number of linear layers.
                - n_convs (int): Number of convolutional layers.
                - save_model_path (str, optional): Path to save the best model. Defaults to None.
                - save_results_path (str): Path to save the optimization results.

        Returns:
            dict: Dictionary containing the validation loss and optimization status.
        """
        global SavedParameters
        global best_loss

        train_data_loader, validation_data_loader = params['dataset'].createTrainAndValidationDataLoader(
            train_batch_size=params['batch_size'], validation_batch_size=params['batch_size'])

        start_train_time = np.datetime64(datetime.now())
        outs = self.train(train_data_loader, validation_data_loader, params)
        end_train_time = np.datetime64(datetime.now())

        torch.cuda.empty_cache()

        SavedParameters.append(outs)
        SavedParameters[-1].update({"learning_rate": params["learning_rate"],
                                    "train_time": str(end_train_time - start_train_time),
                                    "batch_size": params["batch_size"], "hidden_dim": params["hidden_dim"],
                                    "dropout": params["dropout"],
                                    "n_layers": params["n_layers"], "n_convs": params["n_convs"]
                                    })

        if SavedParameters[-1]["validation_loss"] < best_loss:
            if params['save_model_path'] is not None:
                print("New saved model:" + str(SavedParameters[-1]))
                with open(params['save_model_path'], 'wb') as file:
                    print("Saving model in pickle object")
                    pickle.dump(self.model, file)
                    print("Model saved")
                    file.close()
            best_loss = SavedParameters[-1]["validation_loss"]

        SavedParameters = sorted(SavedParameters, key=lambda i: i['validation_loss'], reverse=False)

        with open(params['save_results_path'], 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
            csvfile.close()

        scores = {
            'loss': outs['validation_loss'],
            'status': STATUS_OK
        }

        return scores

    def loadModel(self, load_path, device):
        """
        Loads a pre-trained graph neural network model from a specified path.

        Args:
            load_path (str): Path to the file containing the saved model.
            device (torch.device): Device to load the model onto.

        Returns:
            None
        """
        with open(load_path, 'rb') as file:
            self.model = pickle.load(file)
            self.model.to(device)
            file.close()


class GraphNeuralNetwork(Module):
    def __init__(self, num_node_features, params):
        super().__init__()
        torch.manual_seed(42)
        self.convslen = params['n_convs'] - 1
        self.convs = nn.ModuleList(
            [GCNConv(int(num_node_features * (i + 1)), int(num_node_features * (i + 2))) for i in
             range(params['n_convs'])])

        self.lin = Linear(int(num_node_features * (self.convslen + 2)), params['hidden_dim'],
                          weight_initializer="glorot")
        self.dropout = params['dropout']
        self.len = params['n_layers']
        self.lins = nn.ModuleList(
            [Linear(int(params['hidden_dim'] / (i + 1)), int(params['hidden_dim'] / (i + 2))) for i in
             range(params['n_layers'])])
        self.lin2 = Linear(int(params['hidden_dim'] / (self.len + 1)), 1, weight_initializer="glorot")

    def forward(self, x, edge_index, edge_weight, batch=None):
        edge_index = edge_index.to(torch.long)
        for c in self.convs:
            x = c(x, edge_index, edge_weight=edge_weight)
            x = relu(x)
            x = dropout(x, p=self.dropout)
        x = self.lin(x)
        x = relu(x)
        x = torch.flatten(x, start_dim=1)
        for l in self.lins:
            x = l(x)
            x = relu(x)
            x = dropout(x, p=self.dropout)

        x = global_mean_pool(x, batch)

        x = self.lin2(x)
        x = sigmoid(x)
        return x
