from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

result_file = None


def custom_tokenizer(text):
    return text


def collate_fn(batch):
    return Batch.from_data_list(batch)


def generateGraphFromSequence(sequence, embedding_model, label):
    node_map = {}
    node_features = []
    node_names = []
    edges_with_frequencies = defaultdict(int)

    for i in range(len(sequence)):
        current_string = sequence[i]

        if current_string not in node_map:
            node_index = len(node_map)
            node_map[current_string] = node_index
            node_features.append(embedding_model.getEmbeddedString(current_string))
            node_names.append(current_string)

        if i > 0:
            previous_string = sequence[i - 1]
            src = node_map[previous_string]
            dst = node_map[current_string]
            edges_with_frequencies[(src, dst)] += 1

    edge_index = torch.tensor(np.array(list(edges_with_frequencies.keys())), dtype=torch.int32).t().contiguous()
    edge_attr = torch.tensor(np.array(list(edges_with_frequencies.values())), dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(np.array(node_features), dtype=torch.float)

    # check if there's no edges; in this case a self loop will be added to the only node existing
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.int32)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)

    return Data(x=node_features, y=torch.tensor([label], dtype=torch.float), edge_index=edge_index,
                edge_attr=edge_attr, node_names=node_names)


def generateTensorSequence(sequence, embedding_model, sequence_length, label):
    if len(sequence) > sequence_length:
        embedded_sequences = []
        for current_string in sequence[-sequence_length:]:
            embedded_sequences.append(embedding_model.getEmbeddedString(current_string))
    else:
        embedded_sequences = [[0.0 for _ in range(embedding_model.getEmbeddedStringLength())] for _ in
                              range(sequence_length)]
        for i in range(len(sequence)):
            embedded_sequences[i] = embedding_model.getEmbeddedString(sequence[i])

    return {'x': torch.tensor(np.array(embedded_sequences), dtype=torch.float),
            'y': torch.tensor([label], dtype=torch.float)}


def generateGraphsFromJson(json_example, embedding_model, label):
    graphs = []
    for macro_category in json_example.keys():
        for level_one_action in json_example[macro_category].keys():
            graphs.append(generateGraphFromSequence(json_example[macro_category][level_one_action],
                                                    embedding_model, label))
    return graphs


def generateSequencesFromJson(json_example, embedding_model, sequence_length, label):
    sequences = []
    for macro_category in json_example.keys():
        for level_one_action in json_example[macro_category].keys():
            sequences.append(generateTensorSequence(json_example[macro_category][level_one_action],
                                                    embedding_model, sequence_length, label))
    return sequences


def generateMatrixFromJson(json_example, embedding_model, top_tf_idf_strings):
    matrix = []
    strings_found = 0
    for tf_idf_string in top_tf_idf_strings:
        found = False
        for macro_category in json_example.keys():
            for level_one_action in json_example[macro_category].keys():
                for api_code_string in json_example[macro_category][level_one_action]:
                    if api_code_string == tf_idf_string:
                        strings_found += 1
                        found = True
                        matrix.append(embedding_model.getEmbeddedString(api_code_string))
                        break
                if found:
                    break
            if found:
                break
        if not found:
            matrix.append([0 for _ in range(embedding_model.getEmbeddedStringLength())])
    if strings_found == 0:
        return None
    return matrix


def generate_examples_confidences_csv(predicted_labels, real_labels, confidences, path):
    print("Writing test results in " + path)
    sheet_name = "Test results"
    with pd.ExcelWriter(path) as writer:
        df = pd.DataFrame({
            "Example_index": [i for i in range(1, len(real_labels) + 1)],
            "Predicted_labels": predicted_labels,
            "Real_labels": real_labels,
            "Confidence": confidences,
        })
        df = df.drop(df[df.Predicted_labels != df.Real_labels].index)
        df_sorted = df.sort_values(by='Confidence', ascending=False)
        df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)


def generate_example_prediction_info_xlsx(example_predictions_info, top_n_subgraph, path):
    print("Writing malware prediction info results in " + path)
    columns = ["Example_ID"]
    for i in range(top_n_subgraph):
        columns.extend([
            "NomeFile_sottografo_" + str(i+1),
            "Num_nodi_sottografo_" + str(i+1),
            "Num_archi_sottografo_" + str(i+1),
            "Esiste_ciclo_sottografo_" + str(i+1),
            "Pred_prob_sottografo_" + str(i+1),
        ])
    infos = pd.DataFrame(example_predictions_info, columns=columns)
    with pd.ExcelWriter(path) as writer:
        infos.to_excel(writer, sheet_name="malware subgraphs info", index=False)


def convertTimeDelta(before, after):
    tot = after - before
    d = np.timedelta64(tot, 'D')
    tot -= d
    h = np.timedelta64(tot, 'h')
    tot -= h
    m = np.timedelta64(tot, 'm')
    tot -= m
    s = np.timedelta64(tot, 's')
    tot -= s
    ms = np.timedelta64(tot, 'ms')
    out = str(d) + " " + str(h) + " " + str(m) + " " + str(s) + " " + str(ms)
    return out


def build_result_filename(configuration):
    filename = "results/result"
    for pipeline in [conf for conf in configuration.keys() if "_pipeline" in conf]:
        if configuration[pipeline] == '1':
            filename = filename + "_" + pipeline
    return filename + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".txt"


def open_result_file(path):
    """Open the file in append mode and store it in the global variable."""
    global result_file
    result_file = open(path, "w")


def write_to_result_file(string):
    """Write data to the already-open file."""
    if result_file is not None:
        result_file.write(string + "\n")
    else:
        raise ValueError("File is not open. Call open_file() first.")


def close_result_file():
    """Close the global file if it's open."""
    global result_file
    if result_file is not None:
        result_file.close()
        result_file = None


def get_node_explanation_heatmap(element, min_node_importance, max_node_importance):
    if 'start' in element:
        # edge case
        return 0
    else:
        if 'properties' in element and 'node_importance_value' in element['properties'] and (
                max_node_importance - min_node_importance) != 0:
            return (element['properties']['node_importance_value'] - min_node_importance) / (
                    max_node_importance - min_node_importance)
        return 0


def custom_graph_element_color_mapping(element, min_importance, max_importance):
    if 'start' in element and 'edge_importance_value' in element['properties'] and (
            max_importance - min_importance) != 0:
        # edge case
        normed_value = (element['properties']['edge_importance_value'] - min_importance) / (
                max_importance - min_importance)
    elif 'properties' in element and 'node_importance_value' in element['properties'] and (
            max_importance - min_importance) != 0:
        # node case
        normed_value = (element['properties']['node_importance_value'] - min_importance) / (
                max_importance - min_importance)
    else:
        normed_value = 1
    cmap = plt.get_cmap('BuPu')
    color = cmap(normed_value)
    return f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})"


def get_tensor_min_max(tensor):
    return tensor.min().item(), tensor.max().item()


def count_lists_intersection(list1, list2):
    return sum(1 for item in list1 if item in list2)
