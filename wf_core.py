import warnings

from wf_dataprocessing import mung_data
from wf_ml_evaluation import evaluate_data
from wf_visualization import visualize_data

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    mung_data()
    visualize_data()
    evaluate_data()
