# Normalization function for a cubic patch obtained from preoperative chest CT scans in patients with lung adenocarcinoma
MIN_BOUND = -1200.0
MAX_BOUND = 300.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

# Resizing and normalization
# Cubic patches for lung adenocarcinomas were loaded from test_path
# Data are cubic patches containing CT pixel values
# Cubic patches were obtained using a commercial CT annotation tool (Aview, Coreline Soft)
import numpy as np
import scipy

test_images=[]
for i in range(len(test_path)): 
    a=np.fromfile(test_path[i], dtype=np.int16) # data path
    a1=a.reshape(test_s[i], test_h[i], test_w[i]) # original patch dimension, loaded from a separate dataframe
    resize_factor=[float(50)/test_s[i], float(50)/test_h[i], float(50)/test_w[i]]  
    a2=scipy.ndimage.interpolation.zoom(a1, resize_factor) # resizing
    test_images.append(a2)     
     
test_images1=[]
for i in test_images:
    newimage=normalize(i) #normalization
    test_images1.append(newimage) 

test_images1=np.stack(test_images1,axis=0) 
filenum=len(test_path)
test_images1=test_images1.reshape(filenum, 50, 50, 50, 1) # dimension of data for the model inference

# Model inference
# Details on the model development and validation can be found at https://pubs.rsna.org/doi/10.1148/radiol.2020192764
# Refer to https://github.com/chestrad/survival_prediction for the loss function and dependent libraries

from keras.models import load_model 
model = load_model('model.h5', custom_objects={'loss':surv_likelihood(n_intervals)})
y_pred=model.predict(test_images1,verbose=0) # disease-free survival probabilities at six time intervals are obtained; cumulative product was used for the statistical analysis

# Deep learning model-driven 80 CT features
layer_name = 'global_average_pooling3d_1'
model1= Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
y_pred1=model1.predict(test_images1,verbose=0) # 80 CT features were extracted for the clustering analysis

### Clustering analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as shc   
import matplotlib.pyplot as plt

# PCA
scaler = StandardScaler()
y_pred2 = scaler.fit_transform(y_pred1)    
pca = PCA(10)
pca.fit(y_pred2)
projected = pca.fit_transform(y_pred2)

# PCA: explained variance ratio
plt.figure(figsize=(12,9))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components', fontsize=18)
plt.ylabel('Cumulative explained variance', fontsize=18);
plt.xticks(fontsize=18)
plt.yticks(fontsize=18) 
plt.xticks(range(0,10),
           ["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]) 

# Clustering dendrogram
fig, ax = plt.subplots(figsize=(12, 8))  
ax = shc.dendrogram(shc.linkage(y_pred2, method='ward'))  
plt.title("Hierarchical Clustering Dendrogram", fontsize=18)  
plt.tick_params(\
                axis='x',   
                which='both',                
                bottom='off',  
                top='off',  
                labelbottom='off')
plt.xlabel('Sample', fontsize=18 )
plt.ylabel('Distance', fontsize=18)  
plt.yticks(fontsize=18) 
plt.tight_layout()

# Agglomerative clustering
cluster=AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage='ward')
cluster.fit_predict(y_pred2) 
np.unique(cluster.labels_, return_counts = True)  

# Visualization of the 4 clusters determined using the agglomerative clustering algorithm
plt.figure(figsize=(12,9))
plt.scatter(projected[:, 0], projected[:, 1],
            c=cluster.labels_, edgecolor='none', alpha=0.6, # Grouping of the histopathologic risk factors can be performed in the same manner in the PC biplots
            cmap=plt.cm.get_cmap('Paired', 4))
plt.xlabel('component 1')
plt.ylabel('component 2')
