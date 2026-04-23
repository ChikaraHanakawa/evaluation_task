import numpy
import csv

def read_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        f = csv.DictReader(csv_file)
        data = {"labels": [], "results": [], "probs": []}
        for row in f:
            data["labels"].append(int(row["labels"]))
            data["results"].append(int(row["results"]))
            data["probs"].append(float(row["probs"]))
    return data

def confusion_matrix(input_csv):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index, result in enumerate(input_csv["results"]):
        if result == input_csv["labels"][index]:
            if result == 1:
                tp += 1
            elif result == 0:
                tn += 1
        elif result != input_csv["labels"][index]:
            if result == 1:
                fp += 1
            elif result == 0:
                fn += 1
    return tp, tn, fp, fn

def calc_acc(tp, tn, fp, fn):
    numerator = tp + fp
    dominator = tp + tn + fp + fn
    accuracy = numerator / dominator
    return accuracy

def calc_prec(tp, fp):
    numerator = tp
    dominator = tp + fp
    precision = numerator / dominator
    return precision

def calc_reca(tp, tn):
    numerator = tp
    dominator = tp + tn
    recall = numerator / dominator
    return recall

def calc_f1(precision, recall):
    numerator = 2 * (precision * recall)
    dominator = precision + recall
    f1 = numerator / dominator
    return f1

if __name__ == "__main__":
    data = read_csv("./data.csv")

    tp, tn, fp, fn = confusion_matrix(data)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    accuracy  = calc_acc(tp, tn, fp, fn)
    precision = calc_prec(tp, fp)
    recall    = calc_reca(tp, tn)
    f1        = calc_f1(precision, recall)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
