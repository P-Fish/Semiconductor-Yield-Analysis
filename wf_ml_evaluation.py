import os
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from wf_ml_prediction import evaluate_models
from wf_ml_training import train_classification_models


def split_and_prepare_data(df, target_column, test_size=0.2, standardize=True):
    """
    Split data into training and test sets while ensuring minimum test set size
    and optionally standardizing features.
    """
    # Validate minimum dataset size
    min_required_samples = int(30 / test_size)  # Ensures at least 30 test samples
    if len(df) < min_required_samples:
        raise ValueError(
            f"Dataset too small. Need at least {min_required_samples} samples "
            f"to ensure {30} test samples with test_size={test_size}"
        )

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
    )

    # Verify minimum test set size
    if len(X_test) < 30:
        raise ValueError(
            f"Test set has {len(X_test)} samples. Minimum required is 30. "
            "Please provide more data or adjust test_size."
        )

    # Initialize scaler as None
    scaler = None

    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


def store_split_data(split_data, base_path="data_processed"):
    """
    Store split data components using pickle format.
    """
    try:
        # Create base directory if it doesn't exist
        model_dir = Path(base_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create splits subdirectory for training/test data
        splits_dir = model_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        # Dictionary to store paths
        stored_paths = {}

        # Store training and test features
        for name, data in [
            ('X_train', split_data['X_train']),
            ('X_test', split_data['X_test']),
            ('y_train', split_data['y_train']),
            ('y_test', split_data['y_test'])
        ]:
            # Create path
            file_path = splits_dir / f"{name}.pkl"

            # Save DataFrame using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            stored_paths[name] = str(file_path)

        # Store scaler if present
        if split_data.get('scaler') is not None:
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(split_data['scaler'], f)
            stored_paths['scaler'] = str(scaler_path)

        # Create metadata file with split information
        metadata = {
            'train_samples': len(split_data['X_train']),
            'test_samples': len(split_data['X_test']),
            'features': list(split_data['X_train'].columns),
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'has_scaler': split_data.get('scaler') is not None
        }

        metadata_path = model_dir / "split_metadata.json"
        pd.Series(metadata).to_json(metadata_path)
        stored_paths['metadata'] = str(metadata_path)

        return stored_paths

    except Exception as e:
        raise IOError(f"Error storing split data: {str(e)}")


def load_split_data(base_path="data_processed"):
    """
    Load previously stored split data components.
    """
    try:
        model_dir = Path(base_path)
        splits_dir = model_dir / "splits"

        # Check if directories exist
        if not splits_dir.exists():
            raise FileNotFoundError(f"Splits directory not found at {splits_dir}")

        # Load all components
        split_data = {}

        # Load features and targets
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            file_path = splits_dir / f"{name}.pkl"
            with open(file_path, 'rb') as f:
                split_data[name] = pickle.load(f)

        # Load scaler if it exists
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                split_data['scaler'] = pickle.load(f)

        return split_data

    except Exception as e:
        raise IOError(f"Error loading split data: {str(e)}")


def evaluate_data():
    path_sep = os.path.sep
    pickled_data = (
            'data_processed'
            + path_sep
            + 'serialized'
            + path_sep
            + 'secom_output.pickle'
    )

    with open(pickled_data, 'rb') as f:
        try:
            data_f = pd.DataFrame(pickle.load(f))
            split_data = split_and_prepare_data(data_f, target_column='pass')
            store_split_data(split_data)
            split_data = load_split_data()
            train_classification_models(split_data['X_train'], split_data['y_train'])
            evaluate_models(split_data['X_test'], split_data['y_test'])
        except Exception as err:
            print(err)
            pass