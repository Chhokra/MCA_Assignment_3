import torch
import torch.nn as nn
import pickle
import random 
import operator

from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


mapping = pickle.load(open('Mapping.pkl','rb'))
reverse_mapping = pickle.load(open('Reverse_Mapping.pkl','rb'))

# words = []
# for i in mapping:
# 	words.append(i)

#random_words = random.sample(words,10)
random_words = ['steel', 'brush', 'dessert', 'businesseu', 'luckier', 'sent', 'punitive', 'restrictions', 'surname', 'wyss']
random_indexes = [mapping[i] for i in random_words]


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

class Skip_Gram_W2V(nn.Module):
  
  def __init__(self,vocabulary_size,vector_size):
    super(Skip_Gram_W2V,self).__init__()
    self.hidden_embedding = nn.Embedding(vocabulary_size,vector_size) # This layer contains all the word vectors by the end of the training
  
  def forward(self,center_word_index,context_word_index):
    center_word_tensor = torch.tensor(center_word_index,dtype=torch.long)
    context_word_tensor = torch.tensor(context_word_index,dtype=torch.long)
    center_hidden_embedding = self.hidden_embedding(center_word_tensor)
    context_hidden_embedding = self.hidden_embedding(context_word_tensor) 
    prod_result = torch.matmul(center_hidden_embedding,torch.t(context_hidden_embedding)) 
    return F.logsigmoid(prod_result)

class Skip_Gram_W2V(nn.Module):
  
  def __init__(self,vocabulary_size,vector_size):
    super(Skip_Gram_W2V,self).__init__()
    self.hidden_embedding = nn.Embedding(vocabulary_size,vector_size) # This layer contains all the word vectors by the end of the training
  
  def forward(self,center_word_index,context_word_index):
    center_word_tensor = torch.tensor(center_word_index,dtype=torch.long)
    context_word_tensor = torch.tensor(context_word_index,dtype=torch.long)
    center_hidden_embedding = self.hidden_embedding(center_word_tensor)
    context_hidden_embedding = self.hidden_embedding(context_word_tensor) 
    prod_result = torch.matmul(center_hidden_embedding,torch.t(context_hidden_embedding)) 
    return F.logsigmoid(prod_result)




def find_10_nearest_neighbours(embedding,lookup_index):
	listup = []
	lookup_tensor = torch.tensor([lookup_index],dtype=torch.long)
	lookup_vector = embedding(lookup_tensor)
	similarity_function = nn.CosineSimilarity()
	for i in range(30477):
		comparison_tensor = torch.tensor([i],dtype=torch.long)
		comparison_vector = embedding(comparison_tensor)
		similarity = similarity_function(lookup_vector,comparison_vector)
		listup.append((i,similarity[0].item()))
	
	listup.sort(key=operator.itemgetter(1))
	result_indexes = []
	for i in range(10):
		result_indexes.append(listup[i][0])
	return result_indexes


def plot_model(path):
	model = Skip_Gram_W2V(30477,60)
	model.load_state_dict(torch.load(path))
	vectors = model.hidden_embedding

	embeddings = []
	words = []

	for word in random_words:
		nearest_words = []
		nearest_embeddings = []
		results = find_10_nearest_neighbours(vectors,mapping[word])
		for i in results:
			nearest_words.append(reverse_mapping[i])
			nearest_embeddings.append(vectors(torch.tensor([i],dtype=torch.long)).detach().numpy().flatten())
		embeddings.append(nearest_embeddings)
		words.append(nearest_words)

	embeddings = np.array(embeddings)
	n, m, k = embeddings.shape
	tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embeddings.reshape(n * m, k))).reshape(n, m, 2)

	tsne_plot_similar_words('Similar words for random list', random_words, embeddings_en_2d, words, 0.7,
                        'similar_words.png')





	
	
	


plot_model('../models/5.pth')