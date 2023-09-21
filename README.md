# OANet
We propose OANet using the attention mechanism to predict database performance so that the relationship between Knob and the workload can also be considered

## TRAIN
### Run main.py to train the proposed model. The meaning of paser is as follows. 
<pre>
external      : external matrics (TIME, RATE, WAF, SA)
mode          : kind of neural network ('reshape' is a proposed model, 'single'is a single layer neural network)
hidden_size   : hidden size of the model
group_size    : group size of the model
dot           : Whether to use dot-loss or not
lamb          : application rate of dot loss
lr            : learning rate
act_function  : activation function
epochs        : Number of epochs to run during train step
train         : model goes train mode
eval          : model goes eval mode
</pre>
* #### Training the model
```
python main.py --train --mode {reshape or single } --external {external matrix} --dot --lamb {lamb} --hidden_size {hidden size} --group_size {group size} --epochs {epochs} --lr {learning rate}
```

We used RocksDB in our experiments.
The dataset consists of data for multiple workloads environment.
And each row of the dataset for each workload consists of Knob configuration values and performance(external metrics. e,g,Time,Rate, WAF, SA) values.

## Paper
### This paper received the Excellent Paper Award from the KSC 2021.
Below is link of OANet paper\
[Paper link](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11035616&mark=0&useDate=&ipRange=N&accessgl=Y&language=ko_KR&hasTopBanner=true)
