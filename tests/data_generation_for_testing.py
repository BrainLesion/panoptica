import numpy as np
import pytest


# Test fixtures to generate 2D and 3D test data
@pytest.fixture
def generate_test_data_2d():
    def _generate_test_data(num_ref_instances, num_pred_instances, shape=(100, 100)):
        """
        Generate test data for 2D scenarios.

        Args:
            num_ref_instances (int): Number of reference instances.
            num_pred_instances (int): Number of predicted instances.
            shape (tuple): Shape of the 2D mask.

        Returns:
            tuple: A tuple containing reference masks, predicted masks, reference instances, and predicted instances.
        """
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
        """
        Generate test data for 3D scenarios.

        Args:
            num_ref_instances (int): Number of reference instances.
            num_pred_instances (int): Number of predicted instances.
            shape (tuple): Shape of the 3D mask.

        Returns:
            tuple: A tuple containing reference masks, predicted masks, reference instances, and predicted instances.
        """
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
