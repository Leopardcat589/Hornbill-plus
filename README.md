# Hornbill-plus
About Open-source code for the conditional acceptance paper "Hornbill+: A Wireless Battery-Free Electrochemical IoT Sensing Platform for Agricultural Pesticide Monitoring"

Instructions: First, you need to download the pesticide dataset from the Data source, which is divided into 18 types of pesticides (two enzymes, A and B), 9 pesticide combinations, and different ratios of 3 mixed pesticides. Next, you need to install the corresponding Python environment along with the required libraries and the appropriate CUDA version, and import the downloaded dataset into the Train and Test files. Make sure to place the sequences of enzyme A and enzyme B in the same row. Finally, set the command-line parameters as follows:  
--task_name  
classification  
--is_training  
1  
--root_path  
./EthanolConcentration/  
--model_id  
EthanolConcentration  
--model  
Pyraformer  
--data  
UEA  
--e_layers  
2  
--batch_size  
16  
--d_model  
16  
--d_ff  
32  
--top_k  
3  
--des  
'Exp'  
--itr  
1  
--learning_rate  
0.001  
--train_epochs  
1000  
--patience  
100  
After completing these steps, you can proceed with the deep learning training.
