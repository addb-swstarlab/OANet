# DPPML
We propose DPPML using meta-learning(Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks) mechanism to predict database performance 

## TRAIN
### Run main.py to train the proposed model. The meaning of paser is as follows. 
<pre>
external      : external matrix (TIME, RATE, WAF, SA)
mode          : kind of neural network ('reshape' is a proposed model)
hidden_size   : hidden size of the model
group_size    : group size of the model
dot           : Whether to use dot-loss or not
lamb          : application rate of dot loss
lr            : learning rate
act_function  : activation function
epochs        : epoch number of train
train         : model goes triain mode
eval          : model goes eval mode
</pre>
* #### Training the model
```
python main.py --train --mode {reshape or single } --external {external matrix} --dot --lamb {lamb} --hidden_size {hidden size} --group_size {group size} --epochs {epochs} --lr {learning rate}
```

We used RocksDB in our experiments.
The dataset consists of data for multiple workloads environment.
And each row of the dataset for each workload consists of Knob configuration values and performance values.
