package es.dmr.uimp.clustering

import es.dmr.uimp.clustering.Clustering.elbowSelection
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object BisectingKMeansClusterInvoices {

  def main(args: Array[String]) {

    import Clustering._

    val sparkConf = new SparkConf().setAppName("BisectingClusterInvoices")
    val sc = new SparkContext(sparkConf)

    // load data
    val df = loadData(sc, args(0))
    val featurized = featurizeData(df)
    val filtered = filterData(featurized)
    val dataset = toDataset(filtered)
    dataset.cache()

    dataset.take(5).foreach(println)

    val model = trainModel(dataset)
    // Save model
    model.save(sc, args(1))

    // Save threshold
    val distances = dataset.map(d => distToCentroid(d, model))
    val threshold = distances.top(2000).last

    saveThreshold(threshold, args(2))
  }

  def trainModel(data: RDD[Vector]): BisectingKMeansModel = {

    val models = 1 to 20 map { k =>
      val bkm = new BisectingKMeans()
      bkm.setK(k)
      bkm.run(data)
    }

    val costs = models.map(model => model.computeCost(data))

    val selected = elbowSelection(costs, 0.7)
    System.out.println("Selecting bisecting model: " + models(selected).k)
    models(selected)
  }

  def distToCentroid(datum: Vector, model: BisectingKMeansModel): Double = {
    val centroid = model.clusterCenters(model.predict(datum))
    Vectors.sqdist(datum, centroid)
  }
}