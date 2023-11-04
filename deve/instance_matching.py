import numpy as np
from skimage.measure import label
from scipy.optimize import linear_sum_assignment


def match_instances_with_dice(instance_map, binary_mask, threshold=0.5):
    # Ensure that binary_mask is binary by thresholding it
    binary_mask = (binary_mask > threshold).astype(np.uint8)

    # Label instances in the binary mask, excluding the background (0)
    labeled_mask, num_binary_instances = label(
        binary_mask, background=0, return_num=True
    )

    # Label instances in the instance map, excluding the background (0)
    labeled_instances, num_instance_map_instances = label(
        instance_map, background=0, return_num=True
    )

    # Compute Dice coefficients between instances in binary mask and instance map
    dice_matrix = np.zeros((num_instance_map_instances, num_binary_instances))

    for i in range(1, num_instance_map_instances + 1):
        for j in range(1, num_binary_instances + 1):
            intersection = np.logical_and(labeled_instances == i, labeled_mask == j)
            union = np.logical_or(labeled_instances == i, labeled_mask == j)
            dice_coeff = 2 * intersection.sum() / union.sum() if union.sum() > 0 else 0
            dice_matrix[i - 1][j - 1] = dice_coeff

    # Use linear sum assignment to find the optimal matches
    instance_indices, binary_indices = linear_sum_assignment(-dice_matrix)

    # Initialize the consecutive instances map
    consecutive_instances = np.zeros_like(instance_map, dtype=np.int32)

    # Map the matched instances from the binary mask to the instance map using matching integers
    for instance_idx, binary_idx in zip(instance_indices, binary_indices):
        binary_instance_id = binary_idx + 1
        instance_instance_id = instance_idx + 1
        consecutive_instances[labeled_mask == binary_instance_id] = instance_instance_id

    # Assign negative labels to unmatched instances
    unmatched_instance_indices = set(range(num_instance_map_instances)) - set(
        instance_indices
    )
    unmatched_binary_indices = set(range(num_binary_instances)) - set(binary_indices)

    current_negative_label = -1
    for binary_idx in unmatched_binary_indices:
        current_negative_label -= 1
        consecutive_instances[labeled_mask == (binary_idx + 1)] = current_negative_label

    # Handle instances in the instance map that were not matched
    for instance_idx in unmatched_instance_indices:
        current_negative_label -= 1
        consecutive_instances[
            labeled_instances == (instance_idx + 1)
        ] = current_negative_label

    return consecutive_instances


# Example with the same dimensions:
instance_map = np.array(
    [
        [5, 0, 0, 1, 1, 2, 2, 0, 0],
        [0, 0, 0, 1, 0, 0, 2, 2, 0],
        [0, 0, 3, 3, 0, 0, 0, 2, 2],
        [4, 4, 0, 0, 0, 3, 3, 0, 0],
        [0, 4, 4, 0, 0, 0, 3, 3, 0],
    ]
)

binary_mask = np.array(
    [
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
    ]
)

matched_instances = match_instances_with_dice(instance_map, binary_mask)

print("Consecutive Instances Map:")
print(matched_instances)
