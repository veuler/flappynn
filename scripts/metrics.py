"""Positive-class metrics with an explicit confusion-matrix layout."""
import math

from .dataset import finite_number, require


def classification_metrics(labels, probabilities, threshold=0.5):
    require(len(labels) == len(probabilities) and len(labels) > 0, 'Metrics require aligned nonempty inputs.')
    require(finite_number(threshold) and 0 < threshold < 1, 'Invalid decision threshold.')
    require(all(type(label) is int and label in (0, 1) for label in labels), 'Invalid metric labels.')
    require(all(finite_number(p) and 0 <= p <= 1 for p in probabilities), 'Invalid probabilities.')
    tn = fp = fn = tp = 0
    for label, probability in zip(labels, probabilities):
        if label == 1:
            if probability >= threshold:
                tp += 1
            else:
                fn += 1
        elif probability >= threshold:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'jumpPrecision': precision, 'jumpRecall': recall, 'jumpF1': f1,
        'confusionMatrix': [[tn, fp], [fn, tp]],
        'confusionMatrixLabels': {'rows': ['WAIT', 'JUMP'], 'columns': ['WAIT', 'JUMP']},
        'sampleCount': len(labels), 'jumpCount': tp + fn, 'waitCount': tn + fp,
        'predictedJumps': tp + fp, 'predictionMin': min(probabilities), 'predictionMax': max(probabilities),
        'threshold': threshold, 'accuracy': (tp + tn) / len(labels),
    }


def binary_losses(labels, logits, positive_weight=1.0):
    require(len(labels) == len(logits) and len(labels) > 0, 'Loss requires aligned nonempty inputs.')
    require(all(finite_number(z) for z in logits), 'Non-finite logits.')
    require(finite_number(positive_weight) and positive_weight > 0, 'Invalid positive-class weight.')
    unweighted, weighted = 0.0, 0.0
    for label, logit in zip(labels, logits):
        loss = max(logit, 0) - label * logit + math.log1p(math.exp(-abs(logit)))
        unweighted += loss
        weighted += loss * (positive_weight if label else 1)
    return {'bce': unweighted / len(labels), 'weightedBce': weighted / len(labels)}
