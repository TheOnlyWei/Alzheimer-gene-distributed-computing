import sys
from pyspark.sql import SparkSession
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

"""
For training Gradient Boosted Tree model.

Usage: train_model.py, training (libSVM), iterations (int), model output
"""

def train_gbt(sc, training_filepath, iterations, model_out):
    """
    Trains on all data from training_filepath and outputs the GBT model to
    model_out directory.
    """
    # data is an rdd.
    data = MLUtils.loadLibSVMFile(sc, training_filepath)
    model = GradientBoostedTrees.trainClassifier(data,
                                                 categoricalFeaturesInfo={},
                                                 numIterations=iterations)
    model.save(sc, model_out)


def main():
    # Usage: train_model.py, training (libSVM), iterations (int), model output
    if len(sys.argv) != 4:
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("train_gbt").getOrCreate()
    sc = spark.sparkContext
    train_gbt(sc, sys.argv[1], int(sys.argv[2]), sys.argv[3])

if __name__ == '__main__':
    main()
