from json import JSONDecodeError
from os import listdir
from os.path import isfile, join
import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from utils import custom_tokenizer, write_to_result_file


class JsonDataset:
    """
    A class to handle operations on a dataset of JSON files, including filtering, 
    TF-IDF computation, and sequence extraction.
    """

    def __init__(self, path):
        """
        Initializes the JsonDataset object.

        Args:
            path (str): Path to the dataset directory.
        """
        self.path = path

    def filterDatasetByTFIDF(self, relevant_string_list, save_path):
        """
        Filters the dataset by keeping only the strings present in the relevant string list.

        Args:
            relevant_string_list (list): List of strings to retain in the dataset.
            save_path (str): Path to save the filtered dataset.

        Returns:
            None
        """
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        write_to_result_file('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_path_list:
            try:
                with open(json_path) as f:
                    json_example = json.load(f)
                    filename_parts = os.path.normpath(json_path).split(os.path.sep)
                    file = os.path.join(filename_parts[-2], filename_parts[-1])
                    f.close()
                    for macro_category in json_example.keys():
                        for first_level_value in list(json_example[macro_category]):
                            json_example[macro_category][first_level_value] = [
                                second_level_value
                                for second_level_value in json_example[macro_category][first_level_value]
                                if second_level_value in relevant_string_list
                            ]
                            if len(json_example[macro_category][first_level_value]) == 0:
                                del json_example[macro_category][first_level_value]

                with open(save_path + '/' + file, 'w') as f:
                    json.dump(json_example, f, indent=4)
            except JSONDecodeError:
                os.remove(json_path)
                print("error analyzing json: " + json_path)
                write_to_result_file("error analyzing json: " + json_path)

            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')
                write_to_result_file(str(i) + ' examples computed')

    def filterDatasetByPackages(self, package_list, save_path):
        """
        Filters the dataset by keeping only the strings that contain specific packages.

        Args:
            package_list (list): List of package names to retain in the dataset.
            save_path (str): Path to save the filtered dataset.

        Returns:
            None
        """
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        write_to_result_file('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        if not os.path.exists(save_path):
            os.makedirs(save_path + '/Benign')
            os.makedirs(save_path + '/Malicious')
        for json_path in json_path_list:
            try:
                with open(json_path) as f:
                    json_example = json.load(f)
                    filename_parts = os.path.normpath(json_path).split(os.path.sep)
                    file = os.path.join(filename_parts[-2], filename_parts[-1])
                    f.close()
                    for macro_category in json_example.keys():
                        for first_level_value in list(json_example[macro_category]):
                            json_example[macro_category][first_level_value] = [
                                second_level_value
                                for second_level_value in json_example[macro_category][first_level_value]
                                if any(package in second_level_value for package in package_list)
                            ]
                            if len(json_example[macro_category][first_level_value]) == 0:
                                del json_example[macro_category][first_level_value]

                with open(save_path + '/' + file, 'w') as f:
                    json.dump(json_example, f, indent=4)
            except JSONDecodeError:
                os.remove(json_path)
                print("error analyzing json: " + json_path)
                write_to_result_file("error analyzing json: " + json_path)

            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')
                write_to_result_file(str(i) + ' examples computed')

    def compute_tf_idf(self, top_n=100, save_path=None):
        """
        Computes the TF-IDF scores for the dataset and retrieves the top N words.

        Args:
            top_n (int, optional): Number of top words to retrieve. Defaults to 100.
            save_path (str, optional): Path to save the top words. Defaults to None.

        Returns:
            list: List of top N words based on TF-IDF scores.
        """
        try:
            document_list = self.extract_sequences()

            vectorizer = TfidfVectorizer(stop_words=[], lowercase=False, tokenizer=custom_tokenizer, dtype=np.float32)
            X_train = vectorizer.fit_transform(document_list).toarray()
            feature_names = vectorizer.get_feature_names_out()

            print("Collecting TF-IDF results")
            write_to_result_file("Collecting TF-IDF results")
            tfidf_sums = np.asarray(X_train.sum(axis=0)).flatten()
            tfidf_means = np.asarray(X_train.mean(axis=0)).flatten()

            print("Building results dataframe")
            write_to_result_file("Building results dataframe")
            df = pd.DataFrame({
                'Word': feature_names,
                'Importance (sum)': tfidf_sums,
                'Importance (mean)': tfidf_means
            })
            df_sorted = df.sort_values(by='Importance (mean)', ascending=False)

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(df_sorted.head(top_n)['Word'].to_list(), f)

            print(df_sorted.shape)
            return df_sorted.head(top_n)['Word'].to_list()

        except EOFError:
            print("The file is empty or corrupted.")
            write_to_result_file("The file is empty or corrupted.")
        except pickle.PickleError:
            print("Error while loading the file.")
            write_to_result_file("Error while loading the file.")
        except FileNotFoundError:
            print("The file was not found.")
            write_to_result_file("The file was not found.")

    def extract_sequences(self):
        """
        Extracts sequences from the JSON dataset.

        Returns:
            list: List of sequences extracted from the dataset.
        """
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Extracting sequences from ' + str(len(json_path_list)) + ' files')
        write_to_result_file('Extracting sequences from ' + str(len(json_path_list)) + ' files')

        sequence_list = []
        i = 0
        for json_path in json_path_list:
            with open(json_path) as f:
                json_example = json.load(f)
                f.close()
                for macro_category in json_example.keys():
                    for level_one_action in json_example[macro_category].keys():
                        sequence_list.append(json_example[macro_category][level_one_action])
            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')
                write_to_result_file(str(i) + ' examples computed')

        print(str(len(sequence_list)) + " sequences extracted")
        write_to_result_file(str(len(sequence_list)) + " sequences extracted")
        return sequence_list

    def delete_files_with_empty_properties(self):
        """
        Deletes JSON files with empty properties from the dataset.

        Returns:
            None
        """
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        write_to_result_file('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_path_list:
            with open(json_path) as f:
                json_example = json.load(f)
                f.close()
                if len(json_example['activity']) == 0 and len(json_example['receiver']) == 0 and len(
                        json_example['service']) == 0 and len(json_example['provider']) == 0:
                    os.remove(json_path)
            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')
                write_to_result_file(str(i) + ' examples computed')

    def get_json_path_list(self, label="Benign"):
        """
        Retrieves the list of JSON file paths for a given label.

        Args:
            label (str, optional): Label of the dataset (e.g., "Benign" or "Malicious"). Defaults to "Benign".

        Returns:
            list: List of JSON file paths.
        """
        ds_path_ben = self.path + "/" + label
        return [join(ds_path_ben, f) for f in listdir(ds_path_ben) if isfile(join(ds_path_ben, f))]
