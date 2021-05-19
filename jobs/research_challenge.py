import os
import sys

from pyspark.sql import Window
from pyspark.sql.types import StringType, ArrayType, StructField, StructType, IntegerType, MapType, DoubleType
from pyspark.sql.functions import udf
import pyspark.sql.functions as func
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import re
import random
from visualisation.dependencies import color_scheme

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='research_challenge',
        files=['configs/research_challenge_config.json'])

    log.warn('Running research_challenge analysis...')

    # extracting and transforming the dataset
    data = extract_data(spark)
    data_transformed = transform_data(data, sql_context)

    # top paper authors
    data_transformed = transform_papers_and_authors(data_transformed)
    load_data(data_transformed, "paper_authors")

    # extracting abstracts
    data_transformed = transform_papers_and_abstracts(data_transformed)
    load_data(data_transformed, "paper_abstracts")

    # displaying common words in abstracts
    data_transformed = transform_abstracts_words(data_transformed)
    load_data(data_transformed, "paper_abstracts")

    log.warn('Terminating research_challenge analysis...')

    spark.stop()
    return None


def extract_data(spark):
    base = "../data/open-research"

    sources = [
        "document_parses",
    ]

    sub_sources = [
        "pdf_json",
    ]

    dataframe = None

    for source in sources:
        for sub_source in sub_sources:
            path = f"{base}/{source}/{sub_source}"

            if not os.path.exists(path):
                print(f"Path {path} does not exist", file=sys.stderr)
                continue

            df = (
                spark.read.json(path, schema=generate_cord19_schema(), multiLine=True).withColumn("source",
                                                                                                  func.lit(source))
            )
            if not dataframe:
                dataframe = df
            else:
                dataframe = dataframe.union(df)

    return dataframe


def transform_data(frame, sql_context):
    dt_transformed = frame
    dt_transformed = dt_transformed.fillna("NA")

    return dt_transformed


def transform_papers_and_authors(dataframe):
    df_authors = dataframe.select("paper_id", func.explode("metadata.authors").alias("author")) \
        .select("paper_id", "author.*")
    df_authors.select("first", "middle", "last", "email").where("email <> ''")

    df_authors.show(n=10)

    return dataframe


def transform_papers_and_abstracts(dataframe):
    df_abstracts = (
        dataframe.select("paper_id", func.posexplode("abstract").alias("pos", "value"))
            .select("paper_id", "pos", "value.text")
            .withColumn("ordered_text", func.collect_list("text").over(Window.partitionBy("paper_id").orderBy("pos")))
            .groupBy("paper_id")
            .agg(func.max("ordered_text").alias("sentences"))
            .select("paper_id", func.array_join("sentences", " ").alias("abstract"))
            .withColumn("words", func.size(func.split("abstract", "\s+")))
    )

    df_abstracts.show(n=10)

    return df_abstracts


def transform_abstracts_words(dataframe):
    udf_function_clean = udf(generate_cleaned_abstracts, StringType())
    udf_function_sentiment = udf(generate_sentiment, DoubleType())
    stopwords = set(STOPWORDS)

    dataframe = dataframe.withColumn("clean_abstract", udf_function_clean("abstract"))
    dataframe = dataframe.withColumn("sentiment_abstract", udf_function_sentiment("clean_abstract"))

    dataframe_pd = dataframe.toPandas()
    fig = ff.create_distplot([dataframe_pd["sentiment_abstract"]], ["sentiment_abstract"],
                             colors=[color_scheme.color_400])
    fig.show()

    text = dataframe_pd["clean_abstract"].values
    word_cloud = WordCloud(width=1000, height=500, stopwords=stopwords, background_color="white",
                           max_words=25).generate(
        str(text))
    fig = px.imshow(word_cloud.recolor(color_func=generate_custom_color, random_state=3), binary_compression_level=1,
                    title="Most commonly used words in abstracts")
    fig.show()

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .json("./outputs/research_challenge/" + name, mode='overwrite'))
    return None


def generate_custom_color(word, font_size, position, orientation, random_state=None,
                          **kwargs):
    return random.choice(
        [color_scheme.color_50, color_scheme.color_100, color_scheme.color_200, color_scheme.color_300,
         color_scheme.color_400, color_scheme.color_500, color_scheme.color_600, color_scheme.color_700,
         color_scheme.color_800, color_scheme.color_900])


def generate_cleaned_abstracts(abstract):
    abstract = re.sub('[^a-zA-Z]', ' ', abstract)
    abstract = abstract.lower()
    abstract = abstract.split()
    abstract = ' '.join(abstract)

    return abstract


def generate_sentiment(sentence):
    temp = TextBlob(sentence).sentiment[0]
    if temp == 0.0:
        return 0.0
    else:
        return round(temp, 2)


def generate_cord19_schema():
    author_fields = [
        StructField("first", StringType()),
        StructField("middle", ArrayType(StringType())),
        StructField("last", StringType()),
        StructField("suffix", StringType()),
    ]

    authors_schema = ArrayType(
        StructType(
            author_fields
            + [
                StructField(
                    "affiliation",
                    StructType(
                        [
                            StructField("laboratory", StringType()),
                            StructField("institution", StringType()),
                            StructField(
                                "location",
                                StructType(
                                    [
                                        StructField("addrLine", StringType()),
                                        StructField("country", StringType()),
                                        StructField("postBox", StringType()),
                                        StructField("postCode", StringType()),
                                        StructField("region", StringType()),
                                        StructField("settlement", StringType()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                StructField("email", StringType()),
            ]
        )
    )

    spans_schema = ArrayType(
        StructType(
            [
                StructField("start", IntegerType()),
                StructField("end", IntegerType()),
                StructField("text", StringType()),
                StructField("ref_id", StringType()),
            ]
        )
    )

    section_schema = ArrayType(
        StructType(
            [
                StructField("text", StringType()),
                StructField("cite_spans", spans_schema),
                StructField("ref_spans", spans_schema),
                StructField("eq_spans", spans_schema),
                StructField("section", StringType()),
            ]
        )
    )

    bib_schema = MapType(
        StringType(),
        StructType(
            [
                StructField("ref_id", StringType()),
                StructField("title", StringType()),
                StructField("authors", ArrayType(StructType(author_fields))),
                StructField("year", IntegerType()),
                StructField("venue", StringType()),
                StructField("volume", StringType()),
                StructField("issn", StringType()),
                StructField("pages", StringType()),
                StructField(
                    "other_ids",
                    StructType([StructField("DOI", ArrayType(StringType()))]),
                ),
            ]
        ),
        True,
    )

    ref_schema = MapType(
        StringType(),
        StructType(
            [
                StructField("text", StringType()),
                StructField("latex", StringType()),
                StructField("type", StringType()),
            ]
        ),
    )

    return StructType(
        [
            StructField("paper_id", StringType()),
            StructField(
                "metadata",
                StructType(
                    [
                        StructField("title", StringType()),
                        StructField("authors", authors_schema),
                    ]
                ),
                True,
            ),
            StructField("abstract", section_schema),
            StructField("body_text", section_schema),
            StructField("bib_entries", bib_schema),
            StructField("ref_entries", ref_schema),
            StructField("back_matter", section_schema),
        ]
    )


if __name__ == '__main__':
    main()
