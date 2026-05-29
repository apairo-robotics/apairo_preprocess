"""Compute per-point traversability labels from semantic annotations.

Reads the ``labels`` channel of a RELLIS-3D or SemanticKITTI sequence and
writes a ``trav_label`` channel (uint8, one .npy per scan).

Usage::

    python examples/traversability_from_labels.py /data/Rellis-3D/00000
    python examples/traversability_from_labels.py /data/sequences/00 --dataset semantic_kitti
    python examples/traversability_from_labels.py /data/Rellis-3D/00000 --ids 1 3 10 --overwrite
"""

import argparse

from apairo.dataset.rellis.dataset import Rellis3DDataset
from apairo.dataset.semantic_kitti.dataset import SemanticKittiDataset

from apairo_preprocess.traversability.from_labels import TraversabilityFromLabels

DATASETS = {
    "rellis": Rellis3DDataset,
    "semantic_kitti": SemanticKittiDataset,
}

# Default traversable IDs per dataset when --ids is not provided
DEFAULT_IDS = {
    "rellis": None,           # uses TraversabilityFromLabels built-in defaults
    "semantic_kitti": frozenset({40, 44, 48, 49, 60, 72}),  # road, parking, sidewalk, other-ground, lane-marking, terrain
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seq_dir", help="Sequence directory.")
    p.add_argument("--dataset", choices=DATASETS, default="rellis")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   help="Traversable class IDs. Overrides the dataset defaults.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    traversable_ids = frozenset(args.ids) if args.ids else DEFAULT_IDS[args.dataset]
    preprocessor = TraversabilityFromLabels(traversable_ids=traversable_ids)
    dataset_cls = DATASETS[args.dataset]

    print(f"Dataset  : {dataset_cls.__name__}")
    print(f"Sequence : {args.seq_dir}")
    print(f"IDs      : {preprocessor._trav_ids}")

    dataset_cls.run_preprocess(preprocessor, args.seq_dir, overwrite=args.overwrite)
    print(f"Done — channel '{preprocessor.output_key}' written.")


if __name__ == "__main__":
    main()
