from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

result_file = None


def custom_tokenizer(text):
    """
    Custom tokenizer function.

    Args:
        text (str): Input text to tokenize.

    Returns:
        str: Tokenized text.
    """
    return text


def collate_fn(batch):
    """
    Collates a batch of data into a PyTorch Geometric Batch object.

    Args:
        batch (list): List of data objects.

    Returns:
        Batch: Collated batch object.
    """
    return Batch.from_data_list(batch)


def generateGraphFromSequence(sequence, embedding_model, label):
    """
    Generates a graph from a sequence of strings.

    Args:
        sequence (list): List of strings representing the sequence.
        embedding_model: Model used to generate embeddings for strings.
        label (int): Label for the graph.

    Returns:
        Data: PyTorch Geometric Data object representing the graph.
    """
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

    # Check if there are no edges; add a self-loop if necessary
    if edge_index.numel() == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.int32)
        edge_attr = torch.tensor([[0.0]], dtype=torch.float)

    return Data(x=node_features, y=torch.tensor([label], dtype=torch.float), edge_index=edge_index,
                edge_attr=edge_attr, node_names=node_names)


def generateTensorSequence(sequence, embedding_model, sequence_length, label):
    """
    Generates a tensor sequence from a list of strings.

    Args:
        sequence (list): List of strings representing the sequence.
        embedding_model: Model used to generate embeddings for strings.
        sequence_length (int): Length of the sequence.
        label (int): Label for the sequence.

    Returns:
        dict: Dictionary containing tensors for the sequence and label.
    """
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
    """
    Generates graphs from a JSON example.

    Args:
        json_example (dict): JSON object containing the data.
        embedding_model: Model used to generate embeddings for strings.
        label (int): Label for the graphs.

    Returns:
        list: List of PyTorch Geometric Data objects representing the graphs.
    """
    graphs = []
    for macro_category in json_example.keys():
        for level_one_action in json_example[macro_category].keys():
            graphs.append(generateGraphFromSequence(json_example[macro_category][level_one_action],
                                                    embedding_model, label))
    return graphs


def generateSequencesFromJson(json_example, embedding_model, sequence_length, label):
    """
    Generates tensor sequences from a JSON example.

    Args:
        json_example (dict): JSON object containing the data.
        embedding_model: Model used to generate embeddings for strings.
        sequence_length (int): Length of the sequence.
        label (int): Label for the sequences.

    Returns:
        list: List of dictionaries containing tensors for the sequences and labels.
    """
    sequences = []
    for macro_category in json_example.keys():
        for level_one_action in json_example[macro_category].keys():
            sequences.append(generateTensorSequence(json_example[macro_category][level_one_action],
                                                    embedding_model, sequence_length, label))
    return sequences


def generateMatrixFromJson(json_example, embedding_model, top_tf_idf_strings):
    """
    Generates a matrix from a JSON example using top TF-IDF strings.

    Args:
        json_example (dict): JSON object containing the data.
        embedding_model: Model used to generate embeddings for strings.
        top_tf_idf_strings (list): List of top TF-IDF strings.

    Returns:
        list or None: Matrix of embeddings or None if no strings are found.
    """
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
    """
    Generates a CSV file with example predictions and their confidences.

    Args:
        predicted_labels (list): List of predicted labels.
        real_labels (list): List of real labels.
        confidences (list): List of confidence scores.
        path (str): Path to save the CSV file.

    Returns:
        None
    """
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
    """
    Generates an Excel file with detailed prediction information for examples.

    Args:
        example_predictions_info (list): List of prediction information for examples.
        top_n_subgraph (int): Number of top subgraphs to include.
        path (str): Path to save the Excel file.

    Returns:
        None
    """
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
    """
    Converts a time delta into a formatted string.

    Args:
        before (datetime): Start time.
        after (datetime): End time.

    Returns:
        str: Formatted time delta string.
    """
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
    """
    Builds a result filename based on the configuration.

    Args:
        configuration (dict): Configuration dictionary.

    Returns:
        str: Generated result filename.
    """
    filename = "results/result"
    for pipeline in [conf for conf in configuration.keys() if "_pipeline" in conf]:
        if configuration[pipeline] == '1':
            filename = filename + "_" + pipeline
    return filename + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".txt"


def open_result_file(path):
    """
    Opens a result file in append mode and stores it in a global variable.

    Args:
        path (str): Path to the result file.

    Returns:
        None
    """
    global result_file
    result_file = open(path, "w")


def write_to_result_file(string):
    """
    Writes a string to the already-open result file.

    Args:
        string (str): String to write.

    Returns:
        None
    """
    if result_file is not None:
        result_file.write(string + "\n")
    else:
        raise ValueError("File is not open. Call open_file() first.")


def close_result_file():
    """
    Closes the global result file if it's open.

    Returns:
        None
    """
    global result_file
    if result_file is not None:
        result_file.close()
        result_file = None


def get_node_explanation_heatmap(element, min_node_importance, max_node_importance):
    """
    Computes the heatmap value for a node explanation.

    Args:
        element (dict): Node element.
        min_node_importance (float): Minimum node importance value.
        max_node_importance (float): Maximum node importance value.

    Returns:
        float: Heatmap value for the node.
    """
    if 'start' in element:
        return 0
    else:
        if 'properties' in element and 'node_importance_value' in element['properties'] and (
                max_node_importance - min_node_importance) != 0:
            return (element['properties']['node_importance_value'] - min_node_importance) / (
                    max_node_importance - min_node_importance)
        return 0


def custom_graph_element_color_mapping(element, min_importance, max_importance):
    """
    Maps a graph element to a color based on its importance.

    Args:
        element (dict): Graph element.
        min_importance (float): Minimum importance value.
        max_importance (float): Maximum importance value.

    Returns:
        str: RGB color string.
    """
    if 'start' in element and 'edge_importance_value' in element['properties'] and (
            max_importance - min_importance) != 0:
        normed_value = (element['properties']['edge_importance_value'] - min_importance) / (
                max_importance - min_importance)
    elif 'properties' in element and 'node_importance_value' in element['properties'] and (
            max_importance - min_importance) != 0:
        normed_value = (element['properties']['node_importance_value'] - min_importance) / (
                max_importance - min_importance)
    else:
        normed_value = 1
    cmap = plt.get_cmap('BuPu')
    color = cmap(normed_value)
    return f"rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})"


def get_tensor_min_max(tensor):
    """
    Gets the minimum and maximum values of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        tuple: Minimum and maximum values of the tensor.
    """
    return tensor.min().item(), tensor.max().item()


def count_lists_intersection(list1, list2):
    """
    Counts the number of common elements between two lists.

    Args:
        list1 (list): First list.
        list2 (list): Second list.

    Returns:
        int: Number of common elements.
    """
    return sum(1 for item in list1 if item in list2)
