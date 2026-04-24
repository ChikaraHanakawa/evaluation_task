import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def read_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        f = csv.DictReader(csv_file)
        data = {"labels": [], "results": [], "probs": []}
        for row in f:
            data["labels"].append(int(row["labels"]))
            data["results"].append(int(row["results"]))
            data["probs"].append(float(row["probs"]))
    return data

def confusion_matrix_from_threshold(labels, probs, threshold):
    tp = tn = fp = fn = 0

    for y, p in zip(labels, probs):
        pred = 1 if p >= threshold else 0

        if pred == 1 and y == 1:
            tp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1

    return tp, tn, fp, fn

def auc_trapezoid(fpr, tpr):
    auc = 0.0
    for i in range(1, len(fpr)):
        dx = fpr[i] - fpr[i - 1]
        y = (tpr[i] + tpr[i - 1]) / 2
        auc += dx * y
    return auc

def roc_curve_manual(labels, probs):
    thresholds = sorted(set(probs), reverse=True)

    fpr_list = []
    tpr_list = []

    for th in thresholds:
        tp, tn, fp, fn = confusion_matrix_from_threshold(labels, probs, th)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list

if __name__ == "__main__":
    data = read_csv("./data.csv")

    fpr, tpr = roc_curve_manual(
        labels=data["labels"],
        probs=data["probs"]
    )

    plt.figure()
    plt.plot(fpr, tpr, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (manual)")
    plt.grid(True)
    plt.savefig("./fig/roc.png")
