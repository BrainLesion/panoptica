import numpy as np
import pytest

from panoptica.panoptic_quality import panoptic_quality


@pytest.fixture
def generate_test_data_2d():
    def _generate_test_data(num_ref_instances, num_pred_instances, shape=(100, 100)):
        ref_masks = np.zeros(shape, dtype=np.uint8)
        pred_masks = np.zeros(shape, dtype=np.uint8)

        # Create reference and prediction masks with instances
        ref_instances = []
        pred_instances = []

        for i in range(num_ref_instances):
            x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
            ref_instances.append((x, y))
            ref_masks[x, y] = i + 1

        for i in range(num_pred_instances):
            x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
            pred_instances.append((x, y))
            pred_masks[x, y] = i + 1

        return ref_masks, pred_masks, ref_instances, pred_instances

    return _generate_test_data


@pytest.fixture
def generate_test_data_3d():
    def _generate_test_data(
        num_ref_instances, num_pred_instances, shape=(100, 100, 100)
    ):
        ref_masks = np.zeros(shape, dtype=np.uint8)
        pred_masks = np.zeros(shape, dtype=np.uint8)

        # Create reference and prediction masks with instances
        ref_instances = []
        pred_instances = []

        for i in range(num_ref_instances):
            x, y, z = (
                np.random.randint(0, shape[0]),
                np.random.randint(0, shape[1]),
                np.random.randint(0, shape[2]),
            )
            ref_instances.append((x, y, z))
            ref_masks[x, y, z] = i + 1

        for i in range(num_pred_instances):
            x, y, z = (
                np.random.randint(0, shape[0]),
                np.random.randint(0, shape[1]),
                np.random.randint(0, shape[2]),
            )
            pred_instances.append((x, y, z))
            pred_masks[x, y, z] = i + 1

        return ref_masks, pred_masks, ref_instances, pred_instances

    return _generate_test_data


# Test cases
def test_compute_panoptic_quality_instances_2d_no_overlap(generate_test_data_2d):
    # Test cases for 2D data with no overlap
    ref_masks, pred_masks, _, _ = generate_test_data_2d(0, 0, shape=(100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 1.0
    assert sq == 1.0
    assert rq == 1.0
    assert tp == 0
    assert fp == 0
    assert fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_2d(0, 5, shape=(100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 0.0
    assert sq == 0.0
    assert rq == 0.0
    assert tp == 0
    assert fp == 5
    assert fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_2d(5, 0, shape=(100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 0.0
    assert sq == 0.0
    assert rq == 0.0
    assert tp == 0
    assert fp == 0
    assert fn == 5


def test_compute_panoptic_quality_instances_2d_overlap(generate_test_data_2d):
    # Test case for 2D data with overlap (more meaningful test required)
    ref_masks, pred_masks, _, _ = generate_test_data_2d(5, 5, shape=(100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert 0.0 <= pq <= 1.0
    assert 0.0 <= sq <= 1.0
    assert 0.0 <= rq <= 1.0


def test_compute_panoptic_quality_instances_3d_no_overlap(generate_test_data_3d):
    # Test cases for 3D data with no overlap
    ref_masks, pred_masks, _, _ = generate_test_data_3d(0, 0, shape=(100, 100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 1.0
    assert sq == 1.0
    assert rq == 1.0
    assert tp == 0
    assert fp == 0
    assert fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_3d(0, 5, shape=(100, 100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 0.0
    assert sq == 0.0
    assert rq == 0.0
    assert tp == 0
    assert fp == 5
    assert fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_3d(5, 0, shape=(100, 100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert pq == 0.0
    assert sq == 0.0
    assert rq == 0.0
    assert tp == 0
    assert fp == 0
    assert fn == 5


def test_compute_panoptic_quality_instances_3d_overlap(generate_test_data_3d):
    # Test case for 3D data with overlap (more meaningful test required)
    ref_masks, pred_masks, _, _ = generate_test_data_3d(5, 5, shape=(100, 100, 100))
    pq, sq, rq, tp, fp, fn = panoptic_quality(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        modus="cc",
    )
    assert 0.0 <= pq <= 1.0
    assert 0.0 <= sq <= 1.0
    assert 0.0 <= rq <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
