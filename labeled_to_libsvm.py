import sys
import top_k as tk
from pyspark.sql import SparkSession

"""
Convenience module for extracting and transforming labeled Rosmap file using
gene cluster file and cluster features file into libSVM format.

Usage: to_libsvm.py, rosmap (CSV), gene cluster (CSV), features (CSV), output libSVM

The schema of the features (CSV) file is:
(cluster ID, ...)

The program will work as long as the feature file has the feature cluster IDs
on the first column.
"""

def get_feature_cluster_rdd(clusterfile, featurefile, sc):
    feature_rdd = sc.textFile(featurefile) \
        .map(lambda row: (str(row.split(",")[0]), None))

    result_rdd = tk.get_cluster_file_rdd(clusterfile, ",", sc, None) \
        .join(feature_rdd) \
        .map(lambda row: (row[0], row[1][0])) \
        .flatMap(tk.cluster_split_by_gene)

    return result_rdd

def convert_diagnosis(row):
    # 0 = NCI, 1 = AD
    if row[1] == "1":
        diagnosis = "0"
    else:
        diagnosis = "1"

    return [row[0], diagnosis] + row[2:]

def patient_cluster_key_split(row):
    patient_cluster = row[0].split(";")
    #(patient ID, (diagnosis, [(cluster ID, gene sum)]))
    return (patient_cluster[0], (row[1][0], [(patient_cluster[1], row[1][1])]) )

def to_libSVM_line(row):
    s = row[1][0]
    for v in row[1][1]:
        s = s + " " + v[0] + ":" + str(v[1])
    return s

def cid_to_index(row):
    new_arr = [(str(i+1),tup[1]) for i, tup in enumerate(row[1][1])]
    return (row[0], (row[1][0], new_arr))

def to_libSVM(feature_rdd, output):
    libSVM_rdd = feature_rdd.map(patient_cluster_key_split) \
        .reduceByKey(lambda a, b: (a[0], a[1] + b[1] )) \
        .map(lambda row: ( row[0], (row[1][0], sorted(row[1][1], key=lambda tup: int(tup[0]))) ) ) \
        .map(cid_to_index) \
        .map(to_libSVM_line) \
        .coalesce(1)

    libSVM_rdd.saveAsTextFile(output)

    return libSVM_rdd

def get_libSVM(sc, rosmap, gene_cluster, features, output):
    # sys.argv[1] = rosmap file
    gex_rdd = tk.get_gex_file_rdd(rosmap, ",", sc, None)

    gex_header = sc.broadcast(tk.get_entrez_ids(gex_rdd))
    """
    [patient ID, AD or NCI, genevalue 1, genevalue 2, ...]
    map => (patiend ID, diagnosis, gene value 1), ...
    flatMap => (entrez ID, (patiend ID, diagnosis, gene value 1)), ...
    """
    gex_rdd = gex_rdd.filter(lambda row: row[1] in ["1", "4", "5"]) \
        .map(convert_diagnosis) \
        .map(lambda row: tuple((row[0], row[1], value) for value in row[2:])) \
        .flatMap(lambda row: tuple(zip(gex_header.value, row)) )

    # (gene ID, cluster ID)
    # sys.argv[2] = gene cluster file
    # sys.argv[3] = top k feature clusters
    feature_gene_cluster_rdd = get_feature_cluster_rdd(gene_cluster, features, sc)

    feature_rdd = gex_rdd.join(feature_gene_cluster_rdd) \
        .map(tk.combine_pid_cid) \
        .filter(lambda row: row[1][1] != '') \
        .map(tk.convert_to_float) \
        .reduceByKey(lambda a,b: (a[0], a[1]+b[1]))

    # sys.argv[4] = output libSVM file
    return to_libSVM(feature_rdd, output)

def main():
    # Usage: to_libsvm.py, rosmap (CSV), gene cluster (CSV), features (CSV), output libSVM
    if len(sys.argv) != 5:
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("labeled_to_libsvm").getOrCreate()
    sc = spark.sparkContext
    get_libSVM(sc, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

if __name__ == '__main__':
    main()
