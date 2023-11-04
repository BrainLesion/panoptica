import numpy as np


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def match_instance_maps(reference, prediction, iou_threshold=0.5):
    unique_labels_reference = np.unique(reference)
    unique_labels_prediction = np.unique(prediction)

    matches = []

    for label1 in unique_labels_reference:
        if label1 == 0:  # Skip the background
            continue

        mask1 = reference == label1
        best_iou = 0
        best_match = None

        for label2 in unique_labels_prediction:
            if label2 == 0:  # Skip the background
                continue

            mask2 = prediction == label2
            iou = calculate_iou(mask1, mask2)

            if iou > best_iou:
                best_iou = iou
                best_match = label2

        if best_iou >= iou_threshold:
            matches.append((label1, best_match))

    matched_prediction = np.zeros_like(prediction)
    for match in matches:
        reference_label, prediction_label = match
        matched_prediction[prediction == prediction_label] = reference_label

    return matched_prediction


# Create 10x10 reference and prediction maps with unique instance numbers
reference_map = np.zeros((10, 10), dtype=np.int)
prediction_map = np.zeros((10, 10), dtype=np.int)

# Add True Positives (TP) with unique instance numbers
reference_map[1:4, 1:4] = 1
prediction_map[4:7, 4:7] = 2

# Add False Positives (FP) with unique instance numbers
prediction_map[1:4, 6:9] = 3
prediction_map[6:9, 1:4] = 4

# Add False Negatives (FN) with unique instance numbers
reference_map[6:9, 6:9] = 5

# Specify the IoU threshold
iou_threshold = 0.5

# Call the matching function
matched_prediction = match_instance_maps(reference_map, prediction_map, iou_threshold)

# Print reference, prediction, and matched prediction maps
print("Reference Map:")
print(reference_map)

print("\nPrediction Map:")
print(prediction_map)

print("\nMatched Prediction Map:")
print(matched_prediction)
