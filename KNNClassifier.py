import csv
import math
from collections import Counter

#methods----------------------------------------------------------------------------------

def load_csv(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    return header, data


def euclidean_distance(row1, row2, feature_indices):
    sum_sq = 0
    for idx in feature_indices:
        diff = float(row1[idx]) - float(row2[idx])
        sum_sq += diff ** 2
    return math.sqrt(sum_sq)


def get_neighbors(training_data, test_row, k, feature_indices):
    distances = []

    for train_row in training_data:
        dist = euclidean_distance(train_row, test_row, feature_indices)
        distances.append((train_row, dist))

    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


def predict_classification(training_data, test_row, k, feature_indices, label_index):
    neighbors = get_neighbors(training_data, test_row, k, feature_indices)
    labels = [row[label_index] for row in neighbors]

    vote_counts = Counter(labels)
    prediction = vote_counts.most_common(1)[0][0]

    confidence = vote_counts[prediction] / k

    return prediction, confidence


def compute_accuracy(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)



# Load Data
# ----------------------------

train_header, training_data = load_csv("Training.csv")
test_header, testing_data = load_csv("Testing.csv")

# Identify columns
volume_idx = train_header.index("Volume")
doors_idx = train_header.index("Doors")
style_idx = train_header.index("Style")

feature_indices = [volume_idx, doors_idx]
label_index = style_idx

# Iterate Over K
# ----------------------------

results = []
k_values = range(1, 21)

best_k = 1
best_accuracy = 0

actual_labels = [row[label_index] for row in testing_data]

for k in k_values:
    predictions = []

    for test_row in testing_data:
        prediction, _ = predict_classification(
            training_data, test_row, k,
            feature_indices, label_index
        )
        predictions.append(prediction)

    accuracy = compute_accuracy(actual_labels, predictions)
    results.append([k, accuracy])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Save Accuracy.csv
with open("Accuracy.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["K", "Accuracy"])
    writer.writerows(results)

print(f"Best K found: {best_k}")


# Final Predictions (Best K)
# ----------------------------

final_rows = []

for test_row in testing_data:
    prediction, confidence = predict_classification(
        training_data, test_row, best_k,
        feature_indices, label_index
    )

    final_rows.append(test_row + [prediction, confidence])

# Save updated Testing.csv
with open("Testing.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(test_header + ["Prediction", "Confidence"])
    writer.writerows(final_rows)

print("Files generated:")

