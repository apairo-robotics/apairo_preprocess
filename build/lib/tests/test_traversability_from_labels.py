import numpy as np
import pytest
from apairo.core.sample import Sample

from apairo_preprocess.traversability.from_labels import TraversabilityFromLabels


def _sample(labels):
    return Sample(data={"labels": np.asarray(labels, dtype=np.int32)})


def test_output_dtype():
    out = TraversabilityFromLabels().process(_sample([1, 99]))
    assert out.dtype == np.uint8


def test_output_shape():
    labels = [1, 3, 10, 99, 0]
    out = TraversabilityFromLabels().process(_sample(labels))
    assert out.shape == (len(labels),)


def test_default_ids_all_traversable():
    # All six default RELLIS traversable IDs
    out = TraversabilityFromLabels().process(_sample([1, 3, 10, 23, 31, 33]))
    np.testing.assert_array_equal(out, np.ones(6, dtype=np.uint8))


def test_non_traversable_ids():
    out = TraversabilityFromLabels().process(_sample([0, 2, 5, 100]))
    np.testing.assert_array_equal(out, np.zeros(4, dtype=np.uint8))


def test_mixed_labels():
    out = TraversabilityFromLabels().process(_sample([1, 99, 3, 0]))
    np.testing.assert_array_equal(out, [1, 0, 1, 0])


def test_custom_traversable_ids():
    proc = TraversabilityFromLabels(traversable_ids=frozenset({42, 7}))
    out = proc.process(_sample([42, 7, 1, 3]))
    np.testing.assert_array_equal(out, [1, 1, 0, 0])


def test_stateless_across_calls():
    proc = TraversabilityFromLabels()
    out1 = proc.process(_sample([1, 0]))
    out2 = proc.process(_sample([0, 1]))
    np.testing.assert_array_equal(out1, [1, 0])
    np.testing.assert_array_equal(out2, [0, 1])
