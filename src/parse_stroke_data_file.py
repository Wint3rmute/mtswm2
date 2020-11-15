import numpy as np
import typing

DATA_FILE_NAMES = [
    "inne.txt",
    "ang_prect.txt",
    "ang_prct_2.txt",
    "mi.txt",
    "mi_np.txt",
]

DATA_FILE_PATHS = ["../stroke_data/" + path for path in DATA_FILE_NAMES]


def load_tsv_file(filename: str) -> np.array:
    file = open(filename)
    return np.loadtxt(file)


def convert_tsv_data_to_scikit_compliant(data: np.array) -> np.array:
    # Scikit requires a [ specimen x feature ] data format,
    # so we need to transpose the tsv array
    return data.T


def get_all_data_files() -> typing.List[np.array]:
    return [
        convert_tsv_data_to_scikit_compliant(load_tsv_file(data_file_path))
        for data_file_path in DATA_FILE_PATHS
    ]

def get_dataset_x_y() -> typing.Tuple[np.array]:
    datas = get_all_data_files()
    scikit_compliant_x_matrix = np.concatenate([*datas])
    scikit_compliant_y_matrix = np.concatenate(
        [np.full(data.shape[0], index) for (index, data) in enumerate(datas)]
    )

    return scikit_compliant_x_matrix, scikit_compliant_y_matrix