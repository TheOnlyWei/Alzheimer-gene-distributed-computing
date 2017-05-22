import sys
import time
from pyspark.sql import SparkSession
from math import sqrt

"""
Main program for getting top-k gene cluster as features.
Usage: main.py, rosmap, gene cluster, k (int), output
"""

def get_gex_file_rdd(file_path, delim, sc, minPart):
    result = sc.textFile(file_path,minPartitions=minPart) \
        .map(lambda line: line.split(delim))
    return result

def get_cluster_file_rdd(file_path, delim, sc, minPart):
    result = sc.textFile(file_path, minPartitions=minPart) \
        .map(lambda line: line.split(delim)) \
        .filter(lambda line: line[2] == "Human") \
        .map(lambda line: (line[0], line[4].split(";")))
    return result

def clean_diagnosis(row):
    """
    Finds empty string diagnosis values and replaces them with "NA".
    map: [PID, "", d1, d2, ...] => [PID, "NA", d1, d2, ...]
    """
    row[1] = row[1] if row[1] else "NA"
    return row

def cluster_split_by_gene(row):
    result = []
    for entrez_id in row[1]:
        result.append((entrez_id, row[0]))
    return result

def combine_pid_cid(row):
    new_id = "" + row[1][0][0] + ";" + row[1][1]
    result = (
        new_id, (row[1][0][1], row[1][0][2])
    )
    return result

def clean_gene_cluster(row):
    if len(row[1]) == 1 and row[1][0] == "":
        return False
    return True

def convert_to_float(row):
    converted = float(row[1][1])
    return (row[0], (row[1][0], converted))

def get_ad_nci_size(ad_nci_rdd):
    size_rdd = ad_nci_rdd.map(lambda row: (row[0], 1)) \
        .reduceByKey(lambda a, b: a + b)

    return size_rdd

def get_ad_nci_sum(ad_nci_rdd):
    # sum of each cluster's gene ID sum rows (used in mean calculation).
    # map: (D+CID, GSUM1) + (D+CID, GSUM2) => (CID, GSUM1 + GSUM2) = (CID, SUM)
    ad_nci_rdd = ad_nci_rdd.reduceByKey(lambda a,b: a+b)

    return ad_nci_rdd

def get_mean(sum_rdd, size_rdd):
    mean_rdd = sum_rdd.join(size_rdd) \
        .map(lambda row: (row[0], (row[1][0]/row[1][1]) ))

    return mean_rdd

# Two-pass algorithm
def get_ad_nci_variance(ad_nci_rdd,
                        mean_rdd,
                        size_rdd):

    variance_rdd = ad_nci_rdd.join(mean_rdd) \
        .map(lambda row: (row[0], (row[1][0]-row[1][1])*(row[1][0]-row[1][1]) )) \
        .reduceByKey(lambda a,b: a+b) \
        .join(size_rdd) \
        .map(lambda row: (row[0], row[1][0]/row[1][1]))

    return variance_rdd

def remove_diagnosis(row):
    cluster_id = row[0].split(";")[1]
    return (cluster_id, row[1])

def filter_for_ad(row):
    diagnosis = row[0].split(";")[0]
    return True if diagnosis == "ad" else False

def filter_for_nci(row):
    diagnosis = row[0].split(";")[0]
    return True if diagnosis == "nci" else False

def get_tscore(ad_nci_rdd):
    # ad_rdd: (cluster ID, sum of genes values for row i)
    # nci_rdd: (cluster ID, sum of genes values for row i)
    if ad_nci_rdd.isEmpty():
        print("ERROR: missing ad or nci data, returning None.")
        return None

    size_rdd = get_ad_nci_size(ad_nci_rdd)
    sum_rdd = get_ad_nci_sum(ad_nci_rdd)

    # get mean
    # map: (CID, (SUM, SIZE)) => (CID, SUM/SIZE)
    mean_rdd = get_mean(sum_rdd, size_rdd)

    variance_rdd = get_ad_nci_variance(ad_nci_rdd,
                                       mean_rdd,
                                       size_rdd)

    denominator_rdd = variance_rdd.join(size_rdd) \
        .map(lambda row: (row[0], row[1][0]/row[1][1])) \
        .map(remove_diagnosis) \
        .reduceByKey(lambda a,b: sqrt(a+b)) \
        .filter(lambda row: row[1] != 0.0)

    ad_mean_rdd = mean_rdd.filter(filter_for_ad) \
        .map(remove_diagnosis)
    nci_mean_rdd = mean_rdd.filter(filter_for_nci) \
        .map(remove_diagnosis)

    numerator_rdd = ad_mean_rdd.join(nci_mean_rdd) \
        .map(lambda row: (row[0], row[1][0]-row[1][1]))

    t_value_rdd = numerator_rdd.join(denominator_rdd) \
        .map(lambda row: (row[0], row[1][0]/row[1][1]))

    if t_value_rdd.isEmpty():
        print("ERROR: unable to calculate t-value, all variances are zero. Returning None.")
        return None

    ad_variance_rdd = variance_rdd.filter(filter_for_ad) \
        .map(remove_diagnosis)
    nci_variance_rdd = variance_rdd.filter(filter_for_nci) \
        .map(remove_diagnosis)

    return {"tscore": t_value_rdd,
            "ad_mean": ad_mean_rdd,
            "nci_mean": nci_mean_rdd,
            "ad_variance": ad_variance_rdd,
            "nci_variance": nci_variance_rdd}

def get_top_k_tvalue_rdd(t_value_rdd, k):
    result_rdd = t_value_rdd.map(lambda row: (row[1],row[0])) \
        .sortByKey(ascending=False,keyfunc=lambda k: abs(k)) \
        .zipWithIndex() \
        .filter(lambda row: row[1] < k ) \
        .map(lambda row: (row[0][0], row[0][1]))

    return result_rdd

def get_tscore_above(t_value_rdd, t_score):
    result_rdd = t_value_rdd.filter(lambda row: abs(row[1]) > t_score)

    return result_rdd

def flatten_tuple(tup):
    if type(tup) is not tuple:
        return (tup,)
    if len(tup) == 0:
        return tuple()

    return flatten_tuple(tup[0]) + flatten_tuple(tup[1:])

def combine_diagnosis_cluster(row):
    cluster_id = row[0].split(";")[1]
    if row[1][0] in ["4", "5"]:
        return ("ad;"+cluster_id,row[1][1])
    else:
        return ("nci;"+cluster_id,row[1][1])

def to_string(row):
    s = ""
    for v in row[0:4]:
        s = s + str(v) + ","
    return s + str(sqrt(row[4])) + "," + str(sqrt(row[5]))

def get_entrez_ids(gex_rdd):
    gex_header = gex_rdd.filter(lambda row: row[0] == "PATIENT_ID") \
        .flatMap(lambda row: row)

    gex_header = gex_header.collect()
    del gex_header[0]
    del gex_header[0]

    return gex_header

def main():
    # Usage: main.py, rosmap, gene cluster, k (int), output
    if len(sys.argv) != 5 or not sys.argv[3].isdigit():
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("genex").getOrCreate()
    sc = spark.sparkContext

    seconds_elapsed = time.time()
    gex_rdd = get_gex_file_rdd(sys.argv[1], ",", sc, None)

    gex_header = sc.broadcast(get_entrez_ids(gex_rdd))

    gex_rdd = gex_rdd.filter(lambda row: (row[1] in ["1", "4", "5"])) \
        .map(lambda row: tuple((row[0], row[1], value) for value in row[2:])) \
        .flatMap(lambda row: tuple(zip(gex_header.value, row)) )

    cluster_rdd = get_cluster_file_rdd(sys.argv[2], ",", sc, None) \
        .filter(clean_gene_cluster) \
        .flatMap(cluster_split_by_gene)

    # the filter below gets rid of tuples with missing gene expression values.
    # it is imperative that this be done after the zip step above.
    gex_cluster_rdd = gex_rdd.join(cluster_rdd) \
        .map(combine_pid_cid) \
        .filter(lambda row: row[1][1] != '') \
        .map(convert_to_float) \
        .reduceByKey(lambda a,b: (a[0], a[1]+b[1]))

    # schema: (Diagnosis + ";" + cluster ID, gene value)
    ad_nci_rdd = gex_cluster_rdd.map(combine_diagnosis_cluster)

    tscore_mean_variance_dict = get_tscore(ad_nci_rdd)

    #calculate t-score and get top-k t-score.
    if tscore_mean_variance_dict is not None:
        top_k_rdd = get_top_k_tvalue_rdd(tscore_mean_variance_dict["tscore"], int(sys.argv[3]))

        #output
        seconds_elapsed = time.time() - seconds_elapsed

        top_k_rdd = top_k_rdd.map(lambda row: (row[1], row[0])) \
            .join(tscore_mean_variance_dict["ad_mean"]) \
            .join(tscore_mean_variance_dict["nci_mean"]) \
            .join(tscore_mean_variance_dict["ad_variance"]) \
            .join(tscore_mean_variance_dict["nci_variance"]) \
            .map(lambda row: (row[0],) + flatten_tuple(row[1]) ) \
            .map(to_string)

        top_k_rdd = sc.parallelize((seconds_elapsed,)) \
            .union(top_k_rdd) \
            .coalesce(1)

        top_k_rdd.saveAsTextFile(sys.argv[4])
    else:
        print("RESULT: there are no valid AD or NCI data, or all variances are zero.")
        sys.exit()

if __name__ == '__main__':
    main()
