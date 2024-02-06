# kg_uq
This repo contains python scripts for extracting knowledge graphs and training graphical models on knowledge graphs. 

You need an OpenAI key for extraction. 

To install the required deps with a virtual env:

python -m venv /path/to/kg_uq/venv
source /path/to/kg_uq/venv/bin/activate/
pip -r install /path/to/kg_uq/requirements.txt

To add the venv for use with Jupyter Notebooks: 

pip install --user ipykernel
python -m ipykernel install --user --name=kg_uq_venv
