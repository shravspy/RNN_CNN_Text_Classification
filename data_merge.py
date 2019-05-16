#Script to combine IMDB data and merge them togeather to process


import os

path='./topic-rnn/aclImdb/train/pos'
var=' '
file_name=["positive.txt","negative.txt"]
for name in file_name: 
	file = open(name,'a')
	for filename in os.listdir(path):
    	if filename.endswith(".txt"):
        	with open(path+'/'+filename,'r') as f:
            	var=var+'\n'+'1,'+(f.readlines()[0])
    	else:
        	continue
	file.write(var)
	file.close()