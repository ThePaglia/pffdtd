"""Convert HDF5 (.h5/.hdf5) files into a single CSV file.

The generated CSV is in long format with these columns:
- dataset: HDF5 dataset full path
- index_0, index_1, ...: coordinates within the dataset
- value: dataset value at the coordinates

Examples:
	python h5_csv_converter.py input.h5
	python h5_csv_converter.py input.h5 output.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _iter_dataset_rows(dataset_path: str, array: np.ndarray) -> Iterable[tuple[str, tuple[int, ...], object]]:
	"""Yield rows (dataset_path, index_tuple, value) for an ndarray/scalar."""
	if array.shape == ():
		yield dataset_path, (), array.item()
		return

	for idx in np.ndindex(array.shape):
		value = array[idx]
		if isinstance(value, np.generic):
			value = value.item()
		yield dataset_path, idx, value


def _to_python_scalar(value: object) -> object:
	"""Convert numpy/bytes scalars into CSV-friendly Python scalars."""
	if isinstance(value, bytes):
		try:
			return value.decode("utf-8")
		except UnicodeDecodeError:
			return value.decode("utf-8", errors="replace")
	return value


def convert_h5_to_csv(input_path: Path, output_path: Path) -> None:
	"""Convert all datasets in an HDF5 file to one long-format CSV file."""
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	if input_path.suffix.lower() not in {".h5", ".hdf5"}:
		raise ValueError("Input file must have .h5 or .hdf5 extension")

	max_dims = 0
	rows: list[tuple[str, tuple[int, ...], object]] = []

	with h5py.File(input_path, "r") as h5f:
		def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
			nonlocal max_dims
			if not isinstance(obj, h5py.Dataset):
				return

			data = obj[()]
			array = np.asarray(data)
			max_dims = max(max_dims, array.ndim)
			rows.extend(_iter_dataset_rows(f"/{name}", array))

		h5f.visititems(visitor)

	output_path.parent.mkdir(parents=True, exist_ok=True)

	header = ["dataset", *[f"index_{i}" for i in range(max_dims)], "value"]
	with output_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(header)

		for dataset_path, indices, value in rows:
			padded_indices = list(indices) + [""] * (max_dims - len(indices))
			writer.writerow([dataset_path, *padded_indices, _to_python_scalar(value)])


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert an HDF5 file to CSV")
	parser.add_argument("input_h5", type=Path, help="Path to input .h5/.hdf5 file")
	parser.add_argument(
		"output_csv",
		type=Path,
		nargs="?",
		help="Path to output CSV file (default: same name as input with .csv)",
	)
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	input_path: Path = args.input_h5
	output_path: Path = args.output_csv if args.output_csv else input_path.with_suffix(".csv")

	convert_h5_to_csv(input_path, output_path)
	print(f"CSV written to: {output_path}")


if __name__ == "__main__":
	main()
