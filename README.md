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

## Explanation of the code 
*for others or for my future self :D*

### 1. Preprocessing `preprocess.py`
load the saved triples and create a PyG `data.Data` object that contains `edge_index` (tuples of links), `edge_type` , `num_relation` and `num_nodes` as attributes.

Some preprocessig related steps are done inside the training loop, namely the creation of **batches** and **negative sampling**. 
Even if the full dataset is treated as one batch, the following attributes are stored. 

* `edge_index`: [batchsize + (batchsize* negative sampling ratio), 2]
* `edge_type`: [batchsize + (batchsize* negative sampling ratio)] # negative links have an additional type 
* `pos_edge_index`: [batch_size, 2]
* `neg_edge_index`: [(batchsize* negative sampling ratio), 2]
* `node_ids`: IDs to reference the embeddings for the respective nodes in the model 
* `edge_label`: [batchsize + (batchsize* negative sampling ratio), num_relations + 1]
* `num_nodes`: number of nodes in the batch 
* `num_relations`: number of relation types 

### 2. Model `model.py`
An **Encoder-Decoder Model** obtains embeds nodes into a fixed-dimensional vector space. Different neural networks can be used, such as a GNN (`RelationalEncoder`) or an MLP (`LinearEncoder`).
The node embeddings are randomly initialized and can be optimized jointly during training to find more meaningful embeddings. 
```python
self.node_embeddings = nn.init.uniform_(
    Parameter(torch.empty(data.num_nodes, args.hidden_dim)))
```
The **Decoder** takes the embeddings of the nodes and multiplies them. A relation-specific trainable matrix is also multiplied. The output is a score that is high if a link between the two respective nodes is predicted. 
```python
out = torch.matmul(z_src * z_dst, self.rel_emb)
```
Apply `F.Sigmoid` to have values between zero and 1 for each class.

### 3. Training loop
Iterate through batches, encode the node representations, compute scores with decoder and compare them to the node label to compute the loss function, backpropagate the error and update the parameters of the model.  
```python
z = model.encode(batch)
out = F.sigmoid(model.decode(z, batch))
loss = F.binary_cross_entropy(out, batch.edge_label)
loss.backward()
optimizer.step()
```
The updates are done per batch. The overall loss is summed up. 

### 4. Validation step 
To validate, multiple measures are discussed.

* **AUC** from the classification community
* **H@K, MRR, MR** from the information retrieval community 


