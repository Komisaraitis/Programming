from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, trim, lower, regexp_replace
from pyspark.sql.types import DoubleType
import os


def save_results(total_count, top_10, filename):
    """Сохраняет результаты в файл и выводит в консоль"""
    lines = [
        f"Number of anime in adventure genre with rating > 8.0: {total_count}",
        "",
        "TOP-10 adventure anime:",
        "----------------------------------------------------------------------",
        f"{'Rank':<4} {'Title':<40} {'Rating':<6}",
        "----------------------------------------------------------------------",
    ]

    for i, row in enumerate(top_10, 1):
        title = row["title"] or "No Title"
        score = row["score"] or 0
        display_title = (title[:37] + "...") if len(title) > 40 else title
        lines.append(f"{i:<4} {display_title:<40} {score:<6.2f}")

    for line in lines:
        print(line)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():

    java_home = r"C:\Program Files\Java\jre1.8.0_441"
    if os.path.exists(java_home):
        os.environ["JAVA_HOME"] = java_home

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    spark = (
        SparkSession.builder.appName("AdventureAnimeAnalysis")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .csv("final_animedataset.csv")
    )

    df = df.withColumn("score", col("score").cast(DoubleType()))

    df_clean = df.withColumn(
        "genres_clean", lower(trim(regexp_replace(col("genre"), '"', "")))
    )

    df_exploded = df_clean.withColumn(
        "single_genre", explode(split(col("genres_clean"), ","))
    ).withColumn("single_genre", trim(col("single_genre")))

    adventure_anime = df_exploded.filter(
        (col("single_genre") == "adventure")
        & (col("score") > 8.0)
        & (col("score").isNotNull())
    )

    unique_adventure = adventure_anime.dropDuplicates(["anime_id"])

    total_count = unique_adventure.count()

    top_10 = (
        unique_adventure.select("title", "score", "anime_id")
        .orderBy(col("score").desc())
        .limit(10)
        .collect()
    )

    save_results(total_count, top_10, "spark_results.txt")

    spark.stop()


if __name__ == "__main__":
    main()
