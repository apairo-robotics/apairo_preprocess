"""Ground segmentation from semantic class IDs."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_RELLIS_GROUND_IDS: frozenset[int] = frozenset({1, 3, 10, 23, 31, 33})


class GroundSegmentationFromLabels(FramePreprocessor):
    """Per-point ground/non-ground labels from semantic class IDs.

    Each point is labelled ground (0) if its semantic class ID is in
    ``ground_ids``, non-ground (1) otherwise.

    Output labels: **0 = ground, 1 = non-ground** — same convention as
    ``GroundSegmentationCSF``, ``GroundSegmentationRANSAC``, and
    ``TerraSegGroundSegmentation``.

    Default ground IDs for RELLIS-3D::

        {1: dirt, 3: grass, 10: asphalt, 23: concrete, 31: puddle, 33: mud}

    Args:
        labels_key: Input channel for per-point semantic labels.
        ground_ids: Set of semantic class IDs considered ground.
                    Defaults to the RELLIS-3D ground classes.
        output_key: Override the default channel name ``"ground_labels"``.
    """

    output_key: ClassVar[str] = "ground_labels"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["labels"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["labels"]

    def __init__(
        self,
        labels_key: str = "labels",
        ground_ids: frozenset[int] | None = None,
        output_key: str | None = None,
    ) -> None:
        self._labels_key = labels_key
        self._ground_ids = ground_ids if ground_ids is not None else _RELLIS_GROUND_IDS
        self.input_keys = [labels_key]
        self.sources = [labels_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        labels = np.asarray(sample.data[self._labels_key])
        return (~np.isin(labels, list(self._ground_ids))).astype(np.uint8)
