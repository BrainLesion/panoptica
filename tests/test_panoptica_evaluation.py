


# Test cases
def test_compute_panoptica_quality_instances_2d_no_overlap(generate_test_data_2d):
    """
    Test cases for 2D data with no instance overlap.
    """
    ref_masks, pred_masks, _, _ = generate_test_data_2d(0, 0, shape=(100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_2d(0, 5, shape=(100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 5
    assert result.fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_2d(5, 0, shape=(100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 5


def test_compute_panoptica_quality_instances_2d_overlap(generate_test_data_2d):
    """
    Test cases for 2D data with instance overlap.
    """
    ref_masks, pred_masks, _, _ = generate_test_data_2d(5, 5, shape=(100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert 0.0 <= result.pq <= 1.0
    assert 0.0 <= result.sq <= 1.0
    assert 0.0 <= result.rq <= 1.0


def test_compute_panoptica_quality_instances_3d_no_overlap(generate_test_data_3d):
    """
    Test cases for 3D data with no instance overlap.
    """
    ref_masks, pred_masks, _, _ = generate_test_data_3d(0, 0, shape=(100, 100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_3d(0, 5, shape=(100, 100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 5
    assert result.fn == 0

    ref_masks, pred_masks, _, _ = generate_test_data_3d(5, 0, shape=(100, 100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert result.pq == 0.0
    assert result.sq == 0.0
    assert result.rq == 0.0
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 5


def test_compute_panoptica_quality_instances_3d_overlap(generate_test_data_3d):
    """
    Test cases for 3D data with instance overlap.
    """
    ref_masks, pred_masks, _, _ = generate_test_data_3d(5, 5, shape=(100, 100, 100))
    result = panoptica_evaluation(
        ref_mask=ref_masks,
        pred_mask=pred_masks,
        iou_threshold=0.5,
        modus="cc",
    )
    assert 0.0 <= result.pq <= 1.0
    assert 0.0 <= result.sq <= 1.0
    assert 0.0 <= result.rq <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
