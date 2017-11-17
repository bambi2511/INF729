package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** CHARGER LE DATASET **/
    val df: DataFrame = spark
      .read
      .parquet("/home/bambi/source/INF729/Scala_TP/data/prepared_trainingset")

    // b) nombre de lignes et colonnes
    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    // c) Observer le dataframe: First 20 rows, and all columns :
    df.show()

    // d) Le schema donne le nom et type (string, integer,...) de chaque colonne
    df.printSchema()

    /** TF-IDF **/
    // first stage
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val df_stage_1 = tokenizer.transform(df)
    // df_stage_1.show()

    // second stage
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    //val df_stage_2 = remover.transform(df_stage_1)
    // df_stage_2.show()

    // third stage
    val countVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("counts")
      .setVocabSize(500)
    //.setMinDF(2)


    //val df_stage_3 = countVectorizer.fit(df_stage_2).transform(df_stage_2)
    //df_stage_3.show()

    // fourth stage
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    //val df_stage_4 = idf.fit(df_stage_3).transform(df_stage_3)
    //df_stage_4.show()

    // fifth stage
    val indexerCountry = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    //val df_stage_5 = indexerCountry.fit(df_stage_4).transform((df_stage_4))
    // df_stage_5.show()

    // sixth stage
    val indexerCurrency = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    //val df_stage_6 = indexerCurrency.fit(df_stage_5).transform(df_stage_5)
    // df_stage_6.show()

    /** MODEL **/

    // seventh stage
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    //val df_stage_7 = assembler.transform(df_stage_6)
    //df_stage_7.show()

    // eight stage
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      //    .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    //val lrModel = lr.fit(df_stage_7)

    //println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


    /** PIPELINE **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf, indexerCountry, indexerCurrency, assembler, lr))

    val model = pipeline.fit(df)
    //.transform(df)

    model.write.overwrite().save("/home/bambi/source/INF729/Scala_TP/data/model")

    /** TRAINING AND GRID-SEARCH **/
    //training/testing split
    val Array(df_train, df_test) = df.randomSplit(Array(0.9, 0.1))
    //df_test.show()

    //Cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(countVectorizer.minDF, Array[Double](55, 75, 95))
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


    val cvModel = trainValidationSplit.fit(df_train)
    val df_WithPredictions = cvModel.transform(df_test)

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_WithPredictions))

    df_WithPredictions.groupBy("final_status", "predictions").count.show()
    cvModel.write.overwrite().save("predictions")

    //val cv = new CrossValidator()
    //  .setEstimator(pipeline)
    //  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("final_status"))
    //  .setEstimatorParamMaps(paramGrid)
    //  .setNumFolds(10)

    //val cvModel = cv.fit(df_train)

    //val df_WithPredictions = cvModel.transform(df_test)

    //df_WithPredictions.select("project_id", "name", "final_status", "predictions").show()

    //println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_WithPredictions))

    //df_WithPredictions.groupBy("final_status", "predictions").count().show()

  }
}
