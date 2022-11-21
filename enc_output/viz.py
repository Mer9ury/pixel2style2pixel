import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


n_components = 2
data = np.load('results.npy').reshape(-1,14*512)
model = TSNE(n_components = n_components)

embedded = model.fit_transform(data)
embedded = embedded.reshape(101,9,2)

colors = ['red','blue','yellow','green','pink','purple','navy','orange','black']

for i,color in enumerate(colors):
    plt.scatter(embedded[:,i,0],embedded[:,i,1], color = color, label = i)
# for i,color in enumerate(colors):
#     plt.scatter(embedded[i,:,0],embedded[i,:,1], color = color, label = i)


plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()

plt.savefig('result1.png')