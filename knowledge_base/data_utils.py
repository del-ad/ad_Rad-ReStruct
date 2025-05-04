import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8-sig") as json_file:
        json_knowledge_base = json.load(json_file)

    return json_knowledge_base


def load_csv_file(path: Path, selected_indecies: list[int] = None):
    """
    Load rows from a CSV file as tuples, optionally selecting specific columns.

    Args:
        file_path (str): Path to the CSV file.
        selected_indices (List[int], optional): Column indices to include in the output.
                                                If None, all columns are included.

    Returns:
        List[Tuple]: List of tuples representing rows of the CSV.

    Raises:
        IndexError: If selected_indices contain out-of-bound indices for a row.
    """
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader, 1):
            if selected_indecies is None:
                selected = tuple(row)
            else:
                try:
                    selected = tuple(row[index] for index in selected_indecies)
                except IndexError as e:
                    raise IndexError(
                        f"Row {i} does not have enough columns for selected indices {selected_indecies}. Row: {row}") from e
            data.append(selected)
    return data


def load_csv_file_as_dict(path: Path, key_index: int, selected_indices: List[int] = None) -> Dict[Any, tuple]:
    """
    Load rows from a CSV file into a dictionary, using a specified column as the key,
    and optionally selecting which columns to include in the values.

    Args:
        path (Path): Path to the CSV file.
        key_index (int): Index of the column to use as the dictionary key.
        selected_indices (List[int], optional): Indices of columns to include in the value tuple.
                                                If None, all columns except the key column are used.

    Returns:
        Dict[Any, tuple]: Dictionary mapping key column values to tuples of selected columns.

    Raises:
        IndexError: If selected_indices contain out-of-bound indices for a row.
    """
    data = {}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader, 1):
            try:
                key = row[key_index]
                if selected_indices is None:
                    # All columns except the key_index
                    value = tuple(val for idx, val in enumerate(row) if idx != key_index)
                else:
                    value = tuple(row[idx] for idx in selected_indices)
                data[key] = value
            except IndexError as e:
                raise IndexError(
                    f"Row {i} does not have enough columns. Expected indices: "
                    f"{selected_indices if selected_indices is not None else 'all but key_index=' + str(key_index)}. "
                    f"Row: {row}"
                ) from e
    return data


def _load_csv_file_as_dict2(path: Path, key_index: int, selected_indices: List[int] = None) -> Dict[Any, tuple]:
    """
    Load rows from a CSV file into a dictionary, using a specified column as the key,
    and optionally selecting which columns to include in the values.

    Args:
        path (Path): Path to the CSV file.
        key_index (int): Index of the column to use as the dictionary key.
        selected_indices (List[int], optional): Indices of columns to include in the value tuple.
                                                If None, all columns except the key column are used.

    Returns:
        Dict[Any, tuple]: Dictionary mapping key column values to tuples of selected columns.

    Raises:
        IndexError: If selected_indices contain out-of-bound indices for a row.
    """
    data = {}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader, 1):
            try:
                key = row[key_index]
                if selected_indices is None:
                    # All columns except the key_index
                    value = list(val for idx, val in enumerate(row) if idx != key_index)[0]
                else:
                    value = list(row[idx] for idx in selected_indices)[0]

                if key in data:
                    data[key].append(value)
                else:
                    data[key] = [value]
            except IndexError as e:
                raise IndexError(
                    f"Row {i} does not have enough columns. Expected indices: "
                    f"{selected_indices if selected_indices is not None else 'all but key_index=' + str(key_index)}. "
                    f"Row: {row}"
                ) from e
    return data


"""
Write a python dictionary as a json file to the disk

Args:
    path (Path, optional): Path where the json file should be written. If no path is specified,
    the file will be created in the directory of the script from which the method is called.
    file_name (str): the name of the output json file
    data (dict): the dictionary to be written as json file

Returns:
    None

"""


def write_dict_as_json(file_name: str, data: dict, path: Path = None, ):
    json_file_path = path / file_name if path is not None else file_name

    with open(json_file_path, "w") as f:
        json.dump(data, f)