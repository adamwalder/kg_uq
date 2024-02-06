# Extracting and training heterogeneous GNNs on knowledge graph triples with LLMs and PyTorch Geometric!
The repo contains helper functions for:
  - (1) Extracting knowledge graph triples (and context) from websites.
  - (2) Embeddings the extracted context.
  - (3) Mapping the triples and embedded context to HeteroData() objects for fitting with PyTorch Geometric.

## Dependencies:
The required dependencies are listed in `requirements.txt`. To get started with a python virtual environment:

`python -m venv /path/to/kg_uq/venv`

`source /path/to/kg_uq/venv/bin/activate`

`pip -r install /path/to/kg_uq/requirements.txt`

Note: Python version 3.11.6 was used. 

## Getting Started

To get started take a look in `./kg_notebooks/recipe_notebook_ex.ipynb`. This contains a detailed example on how to extract kg triples from recipe ingredients and directions off https://www.allrecipes.com.  To import the installed virtual env to jupyter notebook, first make sure the venv is active:

`source source /path/to/kg_uq/venv/bin/activate`

Then install `ipykernel` and make the environment available to Jupyter,

`pip install --user ipykernel`

`python -m ipykernel install --user --name=kg_uq_venv`

## PyTorch Geometric

An example of a heterogenous GNN fit to the extracted recipe KG subgraphs is included in `./kg_notebooks/recipe_gnn.ipynb`. Extract more recipes and try out your own GNN fits. 

