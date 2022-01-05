import re
import os
from uuid import uuid4

from sanic import Sanic
from sanic.response import json as json_response, text
from sanic_openapi import openapi3_blueprint
from sanic_openapi.openapi3 import openapi
import multiprocessing

##############################
import json
import pickle
import torch
import early_stopping
import model_autoencoder
import pandas as pd
import numpy as np
import torch.nn as nn
from net import NeoNet


# Load tree transformation
from transform_inference import TransformerInference

data_inference = {}
bao_base = ""
with (open(bao_base + "preprocess_inference_info.pickle", "rb")) as openfile:
    while True:
        try:
            data_inference = pickle.load(openfile)
        except EOFError:
            break
# Load Scaler for query features.
scalerx = None
with (open(bao_base + "scalerx.pkl", "rb")) as openfile:
    while True:
        try:
            scalerx = pickle.load(openfile)
        except EOFError:
            break

pipeline_inverse = None
with (open(bao_base + "pipeline_inverse.pkl", "rb")) as openfile:
    while True:
        try:
            pipeline_inverse = pickle.load(openfile)
        except EOFError:
            break


io_dim_model_data = len(data_inference["tree_transform"].get_pred_index())

# Model hyper-parameters
in_channels_neo_net = 512
ignore_first_aec_data = 18
query_input_size = 2085
query_hidden_inputs = [260, 300]
query_output = 240
tree_units = [512, 256, 128]
tree_units_dense = [64, 32]
tree_activation_tree = nn.LeakyReLU
tree_activation_dense = nn.ReLU

print("query_input_size", query_input_size, query_hidden_inputs)

##################
print("in_channels", in_channels_neo_net + ignore_first_aec_data)
# Create model
model = NeoNet(
    io_dim_model_data,
    query_input_size,
    query_hidden_inputs,
    query_output=query_output,
    tree_units=tree_units,
    tree_units_dense=tree_units_dense,
    activation_tree=tree_activation_tree,
    activation_dense=tree_activation_dense,
)

print(model)

# Loading model parameters.
model_state_dict_path = bao_base + "models_data/best_model.pt"
model.load_state_dict(torch.load(model_state_dict_path, map_location=torch.device('cpu')))
model.eval()

########  Sanic app  ############

app = Sanic("My Hello, world app")

app.blueprint(openapi3_blueprint)


#api_neo_path = "/home/dcasals/source/api_neo/"
api_neo_path = ""
java = "/usr/bin/java"

query_features = f"""
{java} -jar {api_neo_path}sparql-query2vec-0.0.1.jar  query-features \
{api_neo_path}temp/query.sparql  \
{api_neo_path}temp/query_output  \
{api_neo_path}temp/centroids.csv  \
--input-delimiter=ᶶ  \
--output-delimiter=ᶶ  \
--execTimeColumn=2  \
--config-file={api_neo_path}properties.prop
"""

jena_neo_optimizer = "jena-neooptimizer-3.16.0-SNAPSHOT/"
#jena_neo_optimizer = "/home/dcasals/source/jena-neooptimizer-3.16.0-SNAPSHOT/"
tree_features = f"""
{jena_neo_optimizer}bin/tdb2.tdbqueryplan --tdb_tree --explain \
 --loc {api_neo_path}tdb2db/ \
 --queriesFile {api_neo_path}temp/query.sparql \
 --idColumn 0 --queryColumn 1 --delimiter "ᶶ" \
 --outFile {api_neo_path}temp/trees_tdb_features.csv
"""


def extract_algebra_patterns(x):
    """Extend algebra features with filter types and slices"""
    tdb = x["tree_tdb"]
    slicepattern = re.compile("(?:slice|top)ᶲ\d+ᶲ\d+")
    max_slice_limit = 0
    max_slice_start = 0
    has_slice = 0
    for s in slicepattern.findall(tdb):
        vals = s.split("ᶲ")
        if int(vals[2]) > max_slice_limit:
            max_slice_limit = int(vals[2])
            max_slice_start = int(vals[1])
            has_slice = 1
    x["has_slice"] = has_slice
    x["max_slice_limit"] = max_slice_limit
    x["max_slice_start"] = max_slice_start
    filterpattern = re.compile("filter[ᶲ\w+:\d+]+")

    for s in filterpattern.findall(tdb):
        filter_item = s.split("ᶲ")[1:]
        for item in filter_item:
            operator, val = item.split(":")
            if "filter_" + operator in x:
                x["filter_" + operator] += int(val)
            else:
                x["filter_" + operator] = int(val)
    return x


########################

def load_algebra_features():
    temp_algebra_features = pd.read_csv(
        "./temp/query_outputalgebra_features", delimiter="ᶶ"
    )
    temp_algebra_features = temp_algebra_features[
        [
            "query_id", "triple", "bgp", "join", "leftjoin", "union", "filter",
            "graph", "extend", "minus", "path*", "pathN*", "path+", "pathN+", "path?",
            "notoneof", "tolist", "order", "project", "distinct", "reduced", "multi",
            "top", "group", "assign", "sequence", "slice", "treesize"
        ]
    ]
    return temp_algebra_features


def load_graph_pattern_features():
    temp_graph_patterns = pd.read_csv("./temp/query_outputgraph_pattern", delimiter="ᶶ")
    temp_graph_patterns = temp_graph_patterns[
        ["id"] + ["pcs" + str(i) for i in range(25)]
        ]
    return temp_graph_patterns


def get_prediction():

    temp_algebra_features = load_algebra_features()
    temp_graph_patterns = load_graph_pattern_features()

    temp_tree = pd.read_csv("./temp/trees_tdb_features.csv", delimiter="ᶶ", header=None)
    temp_tree.columns = ["id", "query", "trees", "tree_tdb", "json_cardinality"]

    temp_merge = pd.merge(
        temp_algebra_features, temp_graph_patterns, left_on="query_id", right_on="id"
    )
    temp_merge = pd.merge(temp_merge, temp_tree, on="id")

    filter_slice_cols = [
        "filter_lang", "filter_not", "filter_sameTerm", "has_slice", "filter_le",
        "filter_lt", "filter_contains", "filter_subtract", "filter_isIRI",
        "filter_langMatches", "filter_strends", "filter_exists", "filter_regex",
        "filter_str", "filter_or", "max_slice_start", "filter_strstarts",
        "filter_notexists", "filter_ne", "filter_eq", "max_slice_limit", "filter_bound",
        "filter_isBlank", "filter_ge", "filter_gt", "filter_isLiteral"
    ]
    temp_merge = temp_merge.apply(lambda x: extract_algebra_patterns(x), axis=1)

    fly_record = temp_merge.reindex(
        temp_merge.columns.union(filter_slice_cols, sort=False), axis=1, fill_value=0
    )

    list_columns = [
        "assign", "bgp", "distinct", "extend", "filter", "graph", "group", "join",
        "leftjoin", "minus", "multi", "notoneof", "order", "path*", "project",
        "reduced", "sequence", "slice", "tolist", "top", "treesize", "triple", "union",
        "path+", "path?", "pathN*", "pathN+", "pcs0", "pcs1", "pcs2", "pcs3", "pcs4",
        "pcs5", "pcs6", "pcs7", "pcs8", "pcs9", "pcs10", "pcs11", "pcs12", "pcs13",
        "pcs14", "pcs15", "pcs16", "pcs17", "pcs18", "pcs19", "pcs20", "pcs21", "pcs22",
        "pcs23", "pcs24", "filter_bound", "filter_contains", "filter_eq",
        "filter_exists", "filter_ge", "filter_gt", "filter_isBlank", "filter_isIRI",
        "filter_isLiteral", "filter_lang", "filter_langMatches", "filter_le",
        "filter_lt", "filter_ne", "filter_not", "filter_notexists", "filter_or",
        "filter_regex", "filter_sameTerm", "filter_str", "filter_strends",
        "filter_strstarts", "filter_subtract", "has_slice", "max_slice_limit",
        "max_slice_start", "json_cardinality",
    ]
    x_test_query = fly_record[list_columns]
    x_test_tree = fly_record["trees"].values

    ########################

    transformer = TransformerInference(
        model=model,
        tree_transform=data_inference["tree_transform"],
        pipeline_inverse=pipeline_inverse,
        scalerx=scalerx,
        maxcardinality=data_inference["maxcardinality"]
    )
    dataset_val = transformer.prepare(x_test_tree, x_test_query)
    predictions = transformer.getpredictions_info(dataset_val)
    return json_response({"results": predictions})

def extract_query_features_java():
    stream = os.popen(query_features)
    output = stream.read()
    print(output)


def extract_trees_tdb_features_java():
    stream = os.popen(tree_features)
    output = stream.read()
    print(output)

@app.route("/query", methods=["POST"])
@openapi.body(
    {"text/plain": openapi.String(description="Sparql query")},
    description="Raw sparql query",
    required=True,
)
async def test(request):
    query = request.body.decode()
    query = query.replace("\n", " ").replace("\t", " ")
    with open("temp/query.sparql", "w") as f:
        data = "ᶶ".join(["id", "query", "time"]) + "\n"
        query_row = "ᶶ".join([str(uuid4()), query, "1.0"])
        data += query_row
        f.write(data)
    import time
    start = time.time()
    extract_query_features_java()
    extract_trees_tdb_features_java()
    diff = time.time() - start
    print(diff)
    return get_prediction()


if __name__ == "__main__":
    app.run()
