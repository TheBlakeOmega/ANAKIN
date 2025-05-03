import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from utils import write_to_result_file


def generate_boxplot(np_array, plot_title=None, save_path=None):    
    """
    Generates a boxplot from a numpy array.

    Args:
        np_array (numpy.ndarray): Array of data to plot.
        plot_title (str, optional): Title of the plot. Defaults to None.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed. Defaults to None.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(np_array)
    plt.title(plot_title)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def generate_line_plot(np_array, plot_title=None, save_path=None, x_label=None, y_label=None):
    """
    Generates a line plot from a numpy array.

    Args:
        np_array (numpy.ndarray): Array of data to plot.
        plot_title (str, optional): Title of the plot. Defaults to None.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed. Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to None.
        y_label (str, optional): Label for the y-axis. Defaults to None.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 7))
    plt.plot(np_array)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def computeTestResults(real_labels, predicted_labels, prediction_type, computation_time):
    """
    Computes and logs test results, including the confusion matrix and classification report.

    Args:
        real_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.
        prediction_type (str): Type of prediction (e.g., "classification").
        computation_time (float): Time taken for the computation.

    Returns:
        None
    """
    print("Computing confusion matrix")
    write_to_result_file("Computing confusion matrix")
    labels = list(set(real_labels))
    test_confusion_matrix = confusion_matrix(real_labels, predicted_labels, labels=labels)
    print("Computing classification report")
    write_to_result_file("Computing classification report")
    test_classification_report = classification_report(real_labels, predicted_labels, digits=3)
    write_to_result_file("Test confusion matrix:")
    write_to_result_file(str(test_confusion_matrix))
    write_to_result_file("Test classification report:")
    write_to_result_file(str(test_classification_report))
    write_to_result_file("Test computation time")
    write_to_result_file(str(computation_time))
    test_confusion_matrix_plot = ConfusionMatrixDisplay(test_confusion_matrix, display_labels=labels)
    test_confusion_matrix_plot.plot()
    plt.title("Test Confusion Matrix " + prediction_type)
    plt.savefig("test_confusion_matrix.png")
    plt.close()
