import sys
import main as mn
from pyspark.sql import SparkSession
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

"""
For diagnosing a CSV file of patients with the following schema:
(patiend ID, g1, g2, g3, ...)
whhere g1, g2, g3, ... are gene expression values corresponding to an Entrez ID.

Usage: diagnose.py, model input, patient input (CSV), gene cluster, features, diagnosis output

output schema:
(patient ID, 0.0 or 1.0)
where 0.0 means the patient is predicted to not have Alzheimer's Disease, and
1.0 means the patient has Alzheimer's disease.
"""

def get_cluster_to_entrez_rdd(clusterfile, featurefile, sc):
    feature_rdd = sc.textFile(featurefile) \
        .map(lambda row: (str(row.split(",")[0]), None))

    result_rdd = mn.get_cluster_file_rdd(clusterfile, ",", sc, None) \
        .join(feature_rdd) \
        .map(lambda row: (row[0], row[1][0]))

    return result_rdd

def get_entrez_ids(gex_rdd):
    gex_header = gex_rdd.filter(lambda row: row[0] == "PATIENT_ID") \
        .flatMap(lambda row: row)

    gex_header = gex_header.collect()
    del gex_header[0]

    return gex_header

def combine_pid_cid(row):
    new_id = "" + row[1][0][0] + ";" + row[1][1]
    return  (new_id, row[1][0][1])

def split_pid_cid(row):
    pid_cid = row[0].split(";")
    return (pid_cid[0], [(pid_cid[1], row[1])])

def remove_cid(row):
    new_arr = [tup[1] for tup in row[1]]
    return (row[0], new_arr)

def get_feature_cluster_rdd(clusterfile, featurefile, delim, sc):
    feature_rdd = sc.textFile(featurefile) \
        .map(lambda row: (str(row.split(delim)[0]), None))

    result_rdd = mn.get_cluster_file_rdd(clusterfile, delim, sc, None) \
        .join(feature_rdd) \
        .map(lambda row: (row[0], row[1][0])) \
        .flatMap(mn.cluster_split_by_gene)

    return result_rdd


def diagnose(sc, patient_feature_rdd, model_path, out_path):
    model = GradientBoostedTreesModel.load(sc, model_path)
    predictions = model.predict(patient_feature_rdd.map(lambda row: row[1]))
    patient_predictions = patient_feature_rdd.zip(predictions) \
        .map(lambda row: (row[0][0], row[1])) \
        .map(lambda row: row[0] + "," + str(row[1])) \
        .coalesce(1)

    patient_predictions.saveAsTextFile(out_path)

    return patient_predictions

def main():
    # Usage: diagnose.py, model input, input (CSV), gene cluster, features, diagnosis output
    if len(sys.argv) != 6:
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("diagnose").getOrCreate()
    sc = spark.sparkContext

    gex_rdd = sc.textFile(sys.argv[2]) \
        .map(lambda row: row.split(","))

    gex_header = sc.broadcast(get_entrez_ids(gex_rdd))

    gex_rdd = gex_rdd.filter(lambda row: row[0] != "PATIENT_ID") \
        .map(lambda row: tuple((row[0], value) for value in row[1:])) \
        .flatMap(lambda row: tuple(zip(gex_header.value, row)) )

    feature_cluster_rdd = get_feature_cluster_rdd(sys.argv[3], sys.argv[4], ",", sc) \
        .filter(lambda row: row[1] != "")

    patient_feature_rdd = gex_rdd.join(feature_cluster_rdd) \
        .map(combine_pid_cid) \
        .filter(lambda row: row[1] != '') \
        .map(lambda row: (row[0], float(row[1]))) \
        .reduceByKey(lambda a,b: a+b)\
        .map(split_pid_cid) \
        .reduceByKey(lambda a,b: a+b) \
        .map(lambda row: (row[0], sorted(row[1], key=lambda tup: int(tup[0]))) ) \
        .map(remove_cid)

    diagnose(sc, patient_feature_rdd, sys.argv[1], sys.argv[5])

if __name__ == '__main__':
    main()
