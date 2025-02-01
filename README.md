# Facial Recognition through **'Eigenfaces'**
## Project for "Analisi tempo-frequenza e multiscala" (Course 2023/2024) 

<br>
The "Eigenfaces" algorithm works by projecting face images on a subspace of principal components (eigenfaces) and classifying an unknown face by comparing it to known faces based on the minimum distance between projections. The goal is to determine whether the face matches an individual in a dataset. <br>
A dataset of 40 individuals was considered (not available in the repository), each with 10 available images. It was divided into a training set and a test set by partitioning the images into two subsets, selecting k images per subject for training and 10-k for testing.

<br><br>

### Training phase:
The implementation is based on principal component analysis. Each image is treated as a vector in an $R^N$ space, where N is the number of pixels (rows × columns). The principal components are then computed, which correspond to the eigenvectors of the covariance matrix in $R^{N \times N}$ of the centered image vectors $Φ_i=f_i- \hat{f}$ (where $\hat{f}$ is the mean face). The covariance matrix will be $C=\frac{1}{L} ΦΦ^T$. Since the number of images (L) is smaller than the dimensions of the space (N×N), there will be only L significant eigenvectors ${v_1,…,v_L}$, while the remaining eigenvectors will have associated eigenvalues equal to zero. Subsequently, the images are projected onto a smaller subspace generated only by the first L′ selected principal components ${v_1,…,v_L′}$, known as the "eigenfaces" space U.

<br><br>


### Test phase:
The automatic face recognition procedure follows. A new image $f_{new}$​ from the test set is firstly centered $Φ_{new}=f_{new}−\hat{f}$, then $Φ_{new}$​ is projected on the eigenface subspace U, and the Euclidean distance $ϵ$ between $Φ_{new}$​ and its projection $UU^TΦ_{new}$​ is computed. If $ϵ<θ$, the face is recognized.To identify the individual, the projection is compared to the mean projections of training images for each subject $Ω_k$​, assigning the identity based on the minimum distance $∥UU^T Φ_{new} − Ω_k∥^2$. 


<br><br>

### Implementation
An unknown parameter to define is the number of principal components L′. An empirical method to determine it is by observing the plot of the eigenvalues in decreasing order and selecting the values around the "elbow", beyond which the associated eigenvectors will be less significant. 
<br>
![immagine](https://github.com/user-attachments/assets/cc3e48d9-606a-4f2b-a8f3-5a3aaa071fe2)

<br><br>
And example of images representing the approximate faces of the first 4 individuals in the training set using L′=20 principal components, with k=6.
<br>
![immagine](https://github.com/user-attachments/assets/d483bd22-ddd1-44ae-aa26-7c5c90ef2583)

<br><br>
An additional parameter to define is the threshold $θ$, beyond which the subject is considered unknown. To determine it, a grid search can be performed by running the algorithm multiple times and varying the values of the parameters k, $θ$, and L′. By selecting and varying the parameters, it is possible to achieve a good percentage of correctly classified individuals. <br>

The plots representing the (1) fraction of correctly classified test set images as θ varies and the (2) fraction of correctly classified test set images as L′ varies are below:
<br>
![immagine](https://github.com/user-attachments/assets/44ca11ea-145e-4e7b-bffd-f2a90f9b1be3)
![immagine](https://github.com/user-attachments/assets/87d3bb75-0a53-466b-958a-22efd0f4035d)













