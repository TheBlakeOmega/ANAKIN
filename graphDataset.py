import json
import os
import pickle
from utils import generateGraphsFromJson, write_to_result_file, collate_fn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch.utils.data import random_split, Subset
import torch
import shutil


class GraphDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.file_paths = [os.path.join(root, f_name) for f_name in os.listdir(root)]

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            data_obj = pickle.load(f)
        return data_obj

    def get_by_filename(self, file_name):
        for file_path in self.file_paths:
            if file_name in file_path:
                with open(file_path, 'rb') as f:
                    data_obj = pickle.load(f)
                return data_obj
        return None

    def get_filename_by_index(self, idx):
        return self.file_paths[idx].split('\\')[-1]

    def refresh_file_paths(self):
        self.file_paths = [os.path.join(self.root, f_name) for f_name in os.listdir(self.root)]

    def generateGraphDataset(self, json_dataset, embedding_model):
        json_path_list_ben = json_dataset.get_json_path_list(label='Benign')
        json_path_list_mal = json_dataset.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list_ben) + len(json_path_list_mal)))
        write_to_result_file('Number of examples to compute: ' + str(len(json_path_list_ben) + len(json_path_list_mal)))

        example_index = 1
        for json_path in json_path_list_ben:
            self._generateAndSaveGraphsFromJson(json_path, example_index, embedding_model, 0)
            example_index += 1
            if example_index % 100 == 0:
                print(str(example_index) + ' examples computed')
                write_to_result_file(str(example_index) + ' examples computed')

        for json_path in json_path_list_mal:
            self._generateAndSaveGraphsFromJson(json_path, example_index, embedding_model, 1)
            example_index += 1
            if example_index % 100 == 0:
                print(str(example_index) + ' examples computed')
                write_to_result_file(str(example_index) + ' examples computed')

        self.file_paths = [os.path.join(self.root, f_name) for f_name in os.listdir(self.root)]

    def createTrainAndValidationDataLoader(self, train_batch_size=64, validation_batch_size=64, split_ratio=0.2):
        validation_size = int(split_ratio * self.len())
        train_size = self.len() - validation_size
        train_dataset, validation_dataset = random_split(self, [train_size, validation_size])

        return (DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn),
                DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn))

    def createTestDataloader(self, start_sub_string="", batch_size=1):
        if not self._checkIfExistsFileWithPrefix(start_sub_string):
            return None, []
        filtered_indices = []
        for index, file_path in enumerate(self.file_paths):
            if start_sub_string in file_path:
                filtered_indices.append(index)
        filtered_dataset = Subset(self, filtered_indices)
        return (DataLoader(filtered_dataset, batch_size=batch_size, num_workers=16, shuffle=False, collate_fn=collate_fn),
                filtered_indices)

    def _generateAndSaveGraphsFromJson(self, json_path, example_index, embedding_model, graph_label):
        with open(json_path, "rb") as f:
            json_example = json.load(f)
            f.close()
            graphs_list = generateGraphsFromJson(json_example, embedding_model, graph_label)
            graph_index = 1
            if len(graphs_list) == 0:
                with open(self.root + "/_" + str(example_index) + "_graph" + str(graph_index) + ".pkl",
                          "wb") as out_file:
                    empty_graph = Data(y=torch.tensor([graph_label]))
                    pickle.dump(empty_graph, out_file)
            else:
                for graph in graphs_list:
                    with open(self.root + "/_" + str(example_index) + "_graph" + str(graph_index) + ".pkl",
                              "wb") as out_file:
                        pickle.dump(graph, out_file)
                    graph_index += 1

    def _checkIfExistsFileWithPrefix(self, prefix):
        return any(filename.startswith(prefix) for filename in os.listdir(self.root))

    def delete(self):
        file_paths = [os.path.join(self.root, f_name) for f_name in os.listdir(self.root)]
        for f in file_paths:
            os.remove(f)

    def copyExampleInTargetDirectory(self, example_indexes, target_directory_path):
        example_substrings = ["_" + str(example_index) + "_" for example_index in example_indexes]
        for source_path in self.file_paths:
            for example_substring in example_substrings:
                if example_substring in source_path:
                    file_name = source_path.split("\\")[-1]
                    destination_path = target_directory_path + "/" + file_name
                    shutil.copyfile(source_path, destination_path)
                    break
