import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random

# 在生成节点 ID 或标签时，强制使用 UTF-8 编码
from typing import Any

def ensure_utf8(text: Any) -> str:
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', errors='ignore')
    elif isinstance(text, list):
        return [ensure_utf8(x) for x in text]
    elif isinstance(text, dict):
        return {k: ensure_utf8(v) for k, v in text.items()}
    else:
        return str(text)

## 输入数据目录
data_dir = "cureus"
inputdirectory = Path(f"./ollama_knowlage_graph/data_input/{data_dir}")
## 输出的 csv 文件保存目录
out_dir = data_dir
outputdirectory = Path(f"./ollama_knowlage_graph/data_output/{out_dir}")


## Dir PDF Loader
# loader = PyPDFDirectoryLoader(inputdirectory)
## File Loader
# loader = PyPDFLoader("./data/MedicalDocuments/orf-path_health-n1.pdf")
loader = DirectoryLoader(inputdirectory, show_progress=True)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

pages = splitter.split_documents(documents)
print("数据块数量 = ", len(pages))
# print(pages[3].page_content)

from helpers.df_helpers import documents2Dataframe
df = documents2Dataframe(pages)
print(df.shape)
df.head()

## 该函数使用 helpers/prompt 函数从文本中提取关键信息
from helpers.df_helpers import df2Graph
from helpers.df_helpers import graph2Df

## 要使用 LLM 重新生成图形，请将此设置为 True
regenerate = False

if regenerate:
    concepts_list = df2Graph(df, model='qwen/qwen3-vl-4b')
    dfg1 = graph2Df(concepts_list)
    dfg1['node_1'] = dfg1['node_1'].apply(ensure_utf8)
    dfg1['node_2'] = dfg1['node_2'].apply(ensure_utf8)    
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    dfg1.to_csv(outputdirectory / "graph.csv", sep="|", index=False)
    df.to_csv(outputdirectory / "chunks.csv", sep="|", index=False)
else:
    dfg1 = pd.read_csv(outputdirectory / "graph.csv", sep="|")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4
## 将关系的权重增至 4。
## 稍后计算上下文接近度时，我们将把权重设为 1。
print(dfg1.shape)
dfg1.head()

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## 将数据集合转换成节点列表
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # 以块id 为关键字的自连接 在同一文本块之间创建链接。
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # 减少自循环
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## 对边缘进行分组和计数。
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # 边缘掉落 1 次
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


dfg2 = contextual_proximity(dfg1)
dfg2.tail()

dfg = pd.concat([dfg1, dfg2], axis=0)
dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)
dfg

nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
nodes.shape
nodes = [ensure_utf8(node) for node in nodes]  # 强制转换为 UTF-8 字符串

import networkx as nx

G = nx.Graph()

## 为图表添加节点
for node in nodes:
    G.add_node(
        str(node)
    )

## 为图形添加边界
for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4
    )


communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("共同体数量 = ", len(communities))
print(communities)

import seaborn as sns
palette = "hls"

## 现在将这些颜色添加到共同体中，并制作另一个数据集
def colors2Community(communities) -> pd.DataFrame:
    ## 定义调色板
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


colors = colors2Community(communities)
colors

for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

from pyvis.network import Network

graph_output_directory = "./ollama_knowlage_graph/docs/index.html"

net = Network(
    notebook=False,
    # bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    # font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
net.show_buttons(filter_=["physics"])

net.show(graph_output_directory, notebook=False)
