import sys
import top_k as tk
from pyspark.sql import SparkSession

"""
Convenience module for extracting AD and NCI patients from Rosmap CSV file
without the diagnosis so it can be used to verify the workings of a machine
learning model by passing the output file to diagnose.py.

Usage: ad_nci_processor.py, rosmap file, ad out, nci out
"""

def to_string(row):
    s = row[0]
    for v in row[1:]:
        s = s + "," + v
    return s

def output_ad_nci(ad_rdd, nci_rdd, ad_out, nci_out):
    ad_rdd = ad_rdd.map(to_string) \
        .coalesce(1)
    nci_rdd = nci_rdd.map(to_string) \
        .coalesce(1)

    ad_rdd.saveAsTextFile(ad_out)
    nci_rdd.saveAsTextFile(nci_out)

    return {"ad": ad_rdd, "nci": nci_rdd}

def remove_diagnosis(ad_rdd, nci_rdd):
    ad_rdd = ad_rdd.map(lambda row: [row[0]] + row[2:])
    nci_rdd = nci_rdd.map(lambda row: [row[0]] + row[2:])

    return {"ad": ad_rdd, "nci": nci_rdd}

def output_without_diagnosis(ad_rdd, nci_rdd, ad_out, nci_out):
    ad_nci_dict = remove_diagnosis(ad_rdd, nci_rdd)

    ad_rdd = ad_nci_dict["ad"].map(to_string) \
        .coalesce(1)

    nci_rdd = ad_nci_dict["nci"].map(to_string) \
        .coalesce(1)

    ad_rdd.saveAsTextFile(ad_out)
    nci_rdd.saveAsTextFile(nci_out)

    return {"ad": ad_rdd, "nci": nci_rdd}

def main():
    # Usage: get_ad_nci_csv.py, rosmap file, ad out, nci out
    if len(sys.argv) != 4:
        print("ERROR: see README.txt for usage.")
        sys.exit()

    spark = SparkSession.builder.appName("ad_nci_processor").getOrCreate()
    sc = spark.sparkContext

    gex_rdd = tk.get_gex_file_rdd(sys.argv[1], ",", sc, None)

    header_rdd = gex_rdd.filter(lambda row: row[0] == "PATIENT_ID")
    ad_rdd = gex_rdd.filter(lambda row: (row[1] in ["4", "5"]))
    nci_rdd = gex_rdd.filter(lambda row: (row[1] == "1"))

    ad_rdd = header_rdd.union(ad_rdd)
    nci_rdd = header_rdd.union(nci_rdd)

    ad_nci_dict = output_without_diagnosis(ad_rdd, nci_rdd, sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()
