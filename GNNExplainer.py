import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import pickle
import pandas as pd
from yfiles_jupyter_graphs import GraphWidget
from utils import custom_graph_element_color_mapping, get_tensor_min_max, write_to_result_file, count_lists_intersection


class GraphNetworkExplainer:

    def __init__(self, model_to_explain, node_names_path):
        model_to_explain.setModelMode('eval')
        with open(node_names_path, 'rb') as file:
            print("Loading nodes' names from " + node_names_path)
            node_names = pickle.load(file)
            print("nodes' names loaded")
            file.close()

        self.explainer_model = None
        self.model_to_explain = model_to_explain.model
        self.node_indexes_dict = {}
        self.node_importance_dict = {
            "classification": {
                0: {},
                1: {}
            },
            "real_label": {
                0: {},
                1: {}
            }
        }
        self.edge_importance_dict = {
            "classification": {
                0: [[[] for _ in range(len(node_names))] for _ in range(len(node_names))],
                1: [[[] for _ in range(len(node_names))] for _ in range(len(node_names))]
            },
            "real_label": {
                0: [[[] for _ in range(len(node_names))] for _ in range(len(node_names))],
                1: [[[] for _ in range(len(node_names))] for _ in range(len(node_names))]
            }
        }
        self._build_node_indexes_dict(node_names)

    def train_explainer(self, save_path):
        print("Training explainer model")
        self.explainer_model = Explainer(
            model=self.model_to_explain,
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
        )
        with open(save_path, 'wb') as file:
            print("Saving explainer model in pickle object")
            pickle.dump(self.explainer_model, file)
            print("Explainer model saved")
            file.close()

    def load_explainer(self, explainer_load_path, node_importance_save_path=None, edge_importance_save_path=None):
        with open(explainer_load_path, 'rb') as file:
            print("Loading explainer model from pickle object")
            self.explainer_model = pickle.load(file)
            print("Explainer model loaded")
            file.close()
        if node_importance_save_path is not None:
            with open(node_importance_save_path, 'rb') as file:
                print("Loading node importance dictionary from pickle object")
                self.node_importance_dict = pickle.load(file)
                print("Node importance dictionary loaded")
                file.close()
        if edge_importance_save_path is not None:
            with open(edge_importance_save_path, 'rb') as file:
                print("Loading edge importance dictionary from pickle object")
                self.edge_importance_dict = pickle.load(file)
                print("Edge importance dictionary loaded")
                file.close()

    def explain_graphs(self, test_graph_dataset, device, xlsx_save_path, node_importance_save_path,
                       edge_importance_save_path):
        print('Number of examples to compute: ' + str(len(test_graph_dataset)))
        example_index = 0
        for graph in test_graph_dataset:
            if hasattr(graph, 'x') and graph.x is not None:
                self._compute_graph_importances(graph, device)
            example_index += 1
            if example_index % 100 == 0:
                print(str(example_index) + ' examples computed')

        self._compute_importances_means()
        self._save_output_importances_in_pickle(node_importance_save_path, edge_importance_save_path)
        self._save_output_importances_in_xlsx(xlsx_save_path)

    def _build_node_indexes_dict(self, node_names):
        for idx, node_name in enumerate(node_names):
            self.node_indexes_dict[node_name] = idx

    def _compute_graph_importances(self, graph, device):
        graph.to(device)
        with torch.no_grad():
            out = self.model_to_explain(graph.x, graph.edge_index, edge_weight=graph.edge_attr)
        explanation = self.explainer_model(graph.x, graph.edge_index, edge_weight=graph.edge_attr)
        node_mask_with_mean = explanation['node_mask'].mean(dim=1)
        edge_mask = explanation['edge_mask']
        classification = 1 if out.item() > 0.5 else 0
        real_label = graph.y.int().item()

        for idx, node_name in enumerate(graph.node_names):
            if node_name not in self.node_importance_dict["classification"][classification]:
                self.node_importance_dict["classification"][classification][node_name] = [
                    node_mask_with_mean[idx].item()]
            else:
                self.node_importance_dict["classification"][classification][node_name].append(
                    node_mask_with_mean[idx].item())
            if node_name not in self.node_importance_dict["real_label"][real_label]:
                self.node_importance_dict["real_label"][real_label][node_name] = [node_mask_with_mean[idx].item()]
            else:
                self.node_importance_dict["real_label"][real_label][node_name].append(
                    node_mask_with_mean[idx].item())

        current_graph_nodes_indexes = [self.node_indexes_dict[node_name] for node_name in graph.node_names]
        for idx, edge_importance in enumerate(edge_mask):
            row, column = (current_graph_nodes_indexes[graph.edge_index[0][idx].int().item()],
                           current_graph_nodes_indexes[graph.edge_index[1][idx].int().item()])
            self.edge_importance_dict["classification"][classification][row][column].append(edge_importance.item())
            self.edge_importance_dict["real_label"][real_label][row][column].append(edge_importance.item())

    def _compute_importances_means(self):
        print("Calculating node importances' means")
        for label_type in self.node_importance_dict.keys():
            for label in self.node_importance_dict[label_type].keys():
                for node_name in self.node_importance_dict[label_type][label].keys():
                    self.node_importance_dict[label_type][label][node_name] = np.mean(
                        np.array(self.node_importance_dict[label_type][label][node_name]))

        print("Calculating edge importances' means")
        for label_type in self.edge_importance_dict.keys():
            for label in self.edge_importance_dict[label_type].keys():
                for i in range(len(self.edge_importance_dict[label_type][label])):
                    for j in range(len(self.edge_importance_dict[label_type][label][i])):
                        self.edge_importance_dict[label_type][label][i][j] = (
                            np.mean(self.edge_importance_dict[label_type][label][i][j]) if
                            len(self.edge_importance_dict[label_type][label][i][j]) > 0 else -1)
        self.edge_importance_dict["node_indexes_dict"] = self.node_indexes_dict

    def _save_output_importances_in_pickle(self, node_importance_save_path, edge_importance_save_path):
        with open(node_importance_save_path, 'wb') as file:
            print("Saving nodes' importance in pickle object")
            pickle.dump(self.node_importance_dict, file)
            print("nodes' importance saved")
            file.close()
        with open(edge_importance_save_path, 'wb') as file:
            print("Saving edges' importance in pickle object")
            pickle.dump(self.edge_importance_dict, file)
            print("Edges' importance saved")
            file.close()

    def _save_output_importances_in_xlsx(self, xlsx_save_path):
        print("Writing results in " + xlsx_save_path)
        label_types = ["classification", "real_label"]
        node_importance_sheet_name = "Nodes importance"
        start_col = 0
        with pd.ExcelWriter(xlsx_save_path) as writer:
            workbook = writer.book
            worksheet = workbook.create_sheet(node_importance_sheet_name)
            for label_type in label_types:
                for label in self.node_importance_dict[label_type].keys():
                    worksheet.cell(
                        value='Importance by ' + label_type + " for " + ("Benign" if label == 0 else "Malicious"),
                        row=1, column=start_col + 1)
                    df = pd.DataFrame({
                        "Node's string": self.node_importance_dict[label_type][label].keys(),
                        'Importance (mean)': self.node_importance_dict[label_type][label].values()
                    })
                    df_sorted = df.sort_values(by='Importance (mean)', ascending=False)
                    df_sorted.to_excel(writer, sheet_name=node_importance_sheet_name, index=False, startrow=2,
                                       startcol=start_col)
                    start_col += len(df.columns) + 2

    def build_visualization_graph(self, graph, device):
        graph.to(device)
        explanation = self.explainer_model(graph.x, graph.edge_index, edge_weight=graph.edge_attr)
        node_mask_with_mean = explanation['node_mask'].mean(dim=1)
        edge_mask = explanation['edge_mask']
        min_node_importance, max_node_importance = get_tensor_min_max(node_mask_with_mean)

        widget = GraphWidget()
        widget.set_node_color_mapping(
            lambda node: custom_graph_element_color_mapping(node, min_node_importance, max_node_importance))
        widget.nodes = [
            {
                "id": idx,
                "properties": {
                    "label": node_name + " imp: " + f"{node_mask_with_mean[idx].item():.5f}",
                    "node_importance_value": node_mask_with_mean[idx].item()
                }
            } for idx, node_name in enumerate(graph.node_names)
        ]
        widget.edges = [
            {
                "id": idx,
                "start": graph.edge_index[0][idx].item(),
                "end": graph.edge_index[1][idx].item(),
                "properties": {
                    "frequency": graph.edge_attr[idx].item(),
                    "edge_importance_value": edge_importance.item(),
                    "label": "freq: " + str(graph.edge_attr[idx].item()) + " - imp: " + f"{edge_importance:.5f}"
                }
            } for idx, edge_importance in enumerate(edge_mask)
        ]

        widget.directed = True
        widget.get_heat_mapping()
        widget.min_node_importance = min_node_importance
        widget.max_node_importance = max_node_importance

        return widget

    def suggest_graphs_to_show(self, test_graph_dataset):
        benign_top_strings = self._get_top_n_node_names(10, 0)
        malicious_top_strings = self._get_top_n_node_names(10, 1)
        example_index = 0
        print("Examples to compute: " + str(len(test_graph_dataset)))
        benign_examples = {}
        malicious_counter_dict = {}
        for graph_path in test_graph_dataset.file_paths:
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
                if hasattr(graph, 'x') and graph.x is not None:
                    label = graph.y.int().item()
                    if label == 0:
                        benign_examples[graph_path] = count_lists_intersection(graph.node_names, benign_top_strings)
                    elif label == 1:
                        malicious_counter_dict[graph_path] = count_lists_intersection(graph.node_names,
                                                                                      malicious_top_strings)
            example_index += 1
            if example_index % 100 == 0:
                print(str(example_index) + ' examples computed')

        write_to_result_file("\nSuggested Benign graphs to display")
        print("\nSuggested Benign graphs to display")
        self._print_suggested_graphs_to_display(10, benign_examples)
        write_to_result_file("\nSuggested Malicious graphs to display")
        print("\nSuggested Malicious graphs to display")
        self._print_suggested_graphs_to_display(10, malicious_counter_dict)

    def _get_top_n_node_names(self, top_n, label):
        df = pd.DataFrame({
            "Node's string": self.node_importance_dict['classification'][label].keys(),
            'Importance (mean)': self.node_importance_dict['classification'][label].values()
        })
        df = df.sort_values(by='Importance (mean)', ascending=False)
        return df.head(top_n)["Node's string"].tolist()

    def _print_suggested_graphs_to_display(self, top_to_display, counter_dict):
        df = pd.DataFrame({
            "graph path": counter_dict.keys(),
            'relevant nodes': counter_dict.values()
        })
        df = df.sort_values(by='relevant nodes', ascending=False)
        for graph_path in df.head(top_to_display)["graph path"].tolist():
            print(graph_path)
            write_to_result_file(graph_path)
