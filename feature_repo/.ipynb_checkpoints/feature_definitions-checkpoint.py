# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    ConflictPolicy,
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    LabelView,
    Project,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, Json, Map, String, Struct

# Define a project for the feature repo
project = Project(name="feature_repo", description="A project to learn Feast")

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
iris = Entity(name="iris", join_keys=["iris_id"])

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.

#Defining File source
iris_source = FileSource(
    name="iris_source",
    path="data/iris.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
iris_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="iris_features",
    entities=[iris],
   # ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width", dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width", dtype=Float64),
    ],
    online=True,
    source=iris_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "MLOps",
            "owner":"LALLU"},
    enable_validation=True,
    version="latest",
)
