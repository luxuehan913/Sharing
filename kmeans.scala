import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

object KMeansFromHDFS {

  // Function to calculate the Euclidean distance between two points
  def distance(p: Vector, q: Vector): Double = {
    math.sqrt(Vectors.sqdist(p, q))
  }

  // Function to find the closest point in candidates to the point q
  def closestPoint(q: Vector, candidates: Array[Vector]): Vector = {
    candidates.minBy(candidate => distance(q, candidate))
  }

  // Function to add two vectors element-wise
  def add_vec(v1: Vector, v2: Vector): Vector = {
    Vectors.dense((v1.toArray zip v2.toArray).map { case (x, y) => x + y })
  }

  // Function to calculate the average of a cluster of vectors
  def average(cluster: Iterable[Vector]): Vector = {
    if (cluster.isEmpty) return Vectors.zeros(cluster.head.size)
    val sum = cluster.reduce(add_vec)
    Vectors.dense(sum.toArray.map(_ / cluster.size))
  }

  // Function to assign clusters to data points
  def assignClusters(data: RDD[Vector], centroids: Array[Vector]): RDD[(Vector, Int)] = {
    data.map { point =>
      val closestCentroidIndex = centroids.zipWithIndex.minBy { case (centroid, _) => distance(point, centroid) }._2
      (point, closestCentroidIndex)
    }
  }

  // Function to update centroids based on cluster assignments
  def updateCentroids(clusteredData: RDD[(Vector, Int)], k: Int): Array[Vector] = {
    clusteredData
      .groupBy(_._2) // Group by cluster index
      .map { case (clusterId, points) =>
        val vectors = points.map(_._1)
        val avgVector = average(vectors)
        (clusterId, avgVector)
      }
      .collect() // Collect results to the driver
      .sortBy(_._1) // Sort by clusterId
      .map(_._2) // Extract the average vectors
  }

  // K-Means main algorithm
  def kMeans(data: RDD[Vector], k: Int, maxIterations: Int): Array[Vector] = {
    // Randomly initialize centroids
    val centroids = data.takeSample(withReplacement = false, num = k)

    var currentCentroids = centroids
    for (_ <- 0 until maxIterations) {
      val clusteredData = assignClusters(data, currentCentroids)
      val newCentroids = updateCentroids(clusteredData, k)

      // Check for convergence
      if (newCentroids.zip(currentCentroids).forall { case (newC, oldC) => newC.equals(oldC) }) {
        return newCentroids
      }
      currentCentroids = newCentroids
    }
    currentCentroids
  }

  ////////////////////////
  // Main Program Below //
  ////////////////////////

  def main(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder
      .appName("KMeans From HDFS")
      .getOrCreate()

    // Load data from HDFS
    val dataFilePath = "hdfs://localhost:9000/clustering_dataset.txt"
    val rawData: RDD[String] = spark.sparkContext.textFile(dataFilePath)

    // Parse the data into RDD[Vector]
    val pointRDD: RDD[Vector] = rawData.map { line =>
      val values = line.split("\t").map(_.toDouble)
      Vectors.dense(values)
    }

    // Run K-Means with k=3
    val k = 3  // Number of clusters
    val maxIterations = 10
    val centroids = kMeans(pointRDD, k, maxIterations)

    // Output the resulting centroids
    println("Final centroids:")
    centroids.foreach(println)

    // Stop Spark session
    spark.stop()
  }
}

// Call the main function
KMeansFromHDFS.main(Array())
