import sys
import time

import numpy as np
from pyspark.sql import SparkSession


def convertToArray(line, split):
    return np.array([float(x) for x in line.split(split)])


def computeDistance(test, train):
    return int(train[-1]), np.sum((test[:-1] - train[:-1]) ** 2)


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonKNN")\
        .getOrCreate()

    sc = spark.sparkContext

    test_lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    test_RDD = test_lines.map(lambda x: convertToArray(x, split=' ')).cache()

    train_lines = spark.read.text(sys.argv[2]).rdd.map(lambda r: r[0])
    train_RDD = train_lines.map(lambda x: convertToArray(x, split=' ')).cache()

    K = int(sys.argv[3])

    start_time = time.time()

    count = 0
    for test_point in test_RDD.collect():
        
        actual_class = int(test_point[-1])

        distances = train_RDD.map(
            lambda train_point: computeDistance(test_point, train_point))

        k_neigh = sc.parallelize(
            distances.takeOrdered(K, key = lambda p: p[1])).map(
                lambda x: (x[0], 1))

        k_nearest_list = k_neigh.reduceByKey(
            lambda x1, x2: x1 + x2)

        predicted_class = k_nearest_list.takeOrdered(1,
            key = lambda x: -x[1])[0][0]

        if predicted_class == actual_class:
            count += 1

    end_time = time.time()
    
    accuracy = (float(count) / test_RDD.count()) * 100
    time_taken = end_time - start_time

    print ("\nAccuracy: " + str(accuracy) + "%\n")
    print ("\nTime taken: " + str(time_taken) + "\n")   