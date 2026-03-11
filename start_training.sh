#!/bin/bash

TRAINING_FILE=./src/main/resources/training.csv

# Entrenar KMeans
spark-submit --class es.dmr.uimp.clustering.KMeansClusterInvoices \
  --master local[4] \
  target/scala-2.11/anomalyDetection-assembly-1.0.jar \
  ${TRAINING_FILE} ./clustering ./threshold

# Entrenar BisectingKMeans
spark-submit --class es.dmr.uimp.clustering.BisectingKMeansClusterInvoices \
  --master local[4] \
  target/scala-2.11/anomalyDetection-assembly-1.0.jar \
  ${TRAINING_FILE} ./clustering_bisect ./threshold_bisect