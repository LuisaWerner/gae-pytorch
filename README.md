# Ontology-enhanced link prediction on featured knowledge graphs 

** still in progress ** 

Forked from [gae-pytorch](https://github.com/zfjsail/gae-pytorch)
The goal is to do link prediction in an encoder-decoder manner based on the vector representations in the graph data (edges and node features). 
Then the predictions will be used as groundings for ontological knowledge that should be enforced with an residual layer. 

The `examples` directory contains several SOTA link prediction model examples in PyG 
The `gae` directory contains the original GAE code 

## How to run 
1. At first, go to the project directory 
```cd knowledge_GAE```

2. and create the directory to store the data
```mkdir WikiAlumni```

3. load the dataset `raw.pkl` from [here](https://gitlab.inria.fr/luwerner/wikialumni/-/blob/master/WikiAlumniPyG/raw.pkl) and put it in the directory.

4. Install the requirements ```pip install -r requirements.txt```

5. Adapt the parameters in the `conf.json` file

6. Execute `run.py --conf.json`