import sys
from pyspark.sql import SparkSession
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

"""
k-fold cross validation

Usage: gbt_cross_validate.py, data (libSVM), k (int), tree iterations (int)
"""

def gbt_cross_validate(sc, filepath, k, iterations):
    data = MLUtils.loadLibSVMFile(sc, filepath)
    # if  k = 2, then split = [0.5, 0.5]
    split = [1/k] * k
    rdd = sc.emptyRDD()
    rdds = (rdd,) * k

    # Spark will normalize split if the fractions don't add up to one.
    rdds = data.randomSplit(split)

    # rdds_keys and rdds_dict are for choosing training and validation sets.
    rdds_keys = [i for i in range(0,k)]
    rdds_dict = {k: v for k, v in enumerate(rdds)}
    error_accumulator = 0.0
    training = sc.emptyRDD()

    # [true positives, true negatives, false positives, false negatives]
    tp_tn_fp_fn = [0,0,0,0]

    for i in rdds_keys:
        validation = rdds_dict[i]
        total_validation_count = validation.count()

        if total_validation_count == 0:
            print("ERROR: not enough data, random split gave empty validation set.")
            break

        # Create the training set
        for j in rdds_keys:
            if j != i:
                training = training.union(rdds_dict[j])

        model = GradientBoostedTrees.trainClassifier(training,
                                                     categoricalFeaturesInfo={},
                                                     numIterations=iterations)

        predictions = model.predict(validation.map(lambda x: x.features))
        labels_predictions = validation.map(lambda lp: lp.label).zip(predictions)
        predict_error_count = labels_predictions.filter(lambda lp: lp[0] != lp[1]).count()

        error_accumulator = (
            error_accumulator +
            (predict_error_count / total_validation_count)
        )
        # Get true positives
        tp_tn_fp_fn[0] = (
            tp_tn_fp_fn[0] +
            labels_predictions.filter(lambda lp: (lp[0] == 1.0 and lp[1] == 1.0)).count()
        )
        # Get true negatives
        tp_tn_fp_fn[1] = (
            tp_tn_fp_fn[1] +
            labels_predictions.filter(lambda lp: (lp[0] == 0.0 and lp[1] == 0.0)).count()
        )
        # Get false positives
        tp_tn_fp_fn[2] = (
            tp_tn_fp_fn[2] +
            labels_predictions.filter(lambda lp: (lp[0] == 0.0 and lp[1] == 1.0)).count()
        )
        # Get false negatives
        tp_tn_fp_fn[3] = (
            tp_tn_fp_fn[3] +
            labels_predictions.filter(lambda lp: (lp[0] == 1.0 and lp[1] == 0.0)).count()
        )

        training = sc.emptyRDD()

    return {"error": error_accumulator/k, "tp_tn_fp_fn": tp_tn_fp_fn}

def main():
    # Usage: gbt_cross_validate.py, data (libSVM), k (int), iterations (int)
    if len(sys.argv) != 4 or not sys.argv[2].isdigit() or not sys.argv[3].isdigit():
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("gbt_cross_validate").getOrCreate()
    sc = spark.sparkContext
    cv_result = gbt_cross_validate(sc, sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    accuracy = 1 - cv_result["error"]
    print("{k}-fold cross validation accuracy: ".format(k=sys.argv[2]), accuracy)
    print("tp, tn, fp, fn: ", cv_result["tp_tn_fp_fn"])

if __name__ == '__main__':
    main()
