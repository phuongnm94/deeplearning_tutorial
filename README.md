## Overview of Deeplearning system 
1. **DataLoader**: load data from files and split it into minibatch. this process is supported by pytorch_lightning library (https://pytorch-lightning.readthedocs.io/en/stable/)
   1. generate word vocab to convert input sentence to the number (check this [data_loarder source file](./src/0.data_loader_sample.ipynb) and take a look to the [dataset example](./src/1.emotion_dataset.ipynb) )
   2. generate the label vocab to convert output label to the number (check [vocab_generator source file](./src/2.emotion_vocab_generation.ipynb))
   3. design function add padding to the input sentence if it necessary  
2. **Model architecture**: 
   1. implement all the important functions (check the bellow image)
      1. learn the FFN network via an examples: 
         1. construct NN not using library [example1](./src/SimpleNN/1.FFN_no_lib.ipynb) 
         2. and using `torch.nn`, `torch.optim` libraries [example2](./src/SimpleNN/2.FFN_torch_lib.ipynb) 
         3. Do [exercise 4](./src/SimpleNN/3.ex4.FFN.ipynb)
   2. check the optimizer function (Adam/SGD/Adagrad/..) and learning rate values
   3. write log or tensorboard  

3. **Exercises**
   1. Fix the error in the this code [ex1](src/0.ex1.ipynb). Why it is error?   
   2. Design a TextPreprocessor for data_loader  [ex2](src/0.ex2.ipynb). 
   3. Code a DataLoader from `emotion` dataset  [ex3](src/2.ex_dataloader.ipynb). 
   4. Design a FFN model to predict next 10 numbers of a magic array [ex4](./src/SimpleNN/3.ex4.FFN.ipynb)

![overview](img/overview_dl.drawio.png)

##  Python ENV 

```cmd
    conda create --prefix=./env_py38  python=3.8
    conda activate ./env_py38 
    pip install -r requirements.txt
```

