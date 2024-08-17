# 046211-Fine-tuning-effect-on-attention-visualization
Comparing the attention visualization of pre-trained ViT classifications on sports images before and after fine-tuning.

# Project documentation
## Topics
* Introduction
  * LRP
  * LoRA & DoRA
* Project goal
* Method
* Experiments and results
* Conclusions
* Future work
* How to run
* Ethics Statement
* Credits

## Introduction
## LRP (Layer-wise Relevance Propagation)
For a machine learning model to generalize well, one needs to ensure that its decisions are supported by meaningful patterns in the input data. A prerequisite is however for the model to be able to explain itself, e.g. by highlighting which input features it uses to support its prediction. Layer-wise Relevance Propagation (LRP) is a technique that brings such explainability and scales to potentially highly complex deep neural networks. It operates by propagating the prediction backward in the neural network, using a set of purposely designed propagation rules.
A simple method, Taylor Decomposition, produces explanations by performing a Taylor expansion of the prediction f(x) at some nearby reference point x&#771;:

![image](https://github.com/user-attachments/assets/1080a40a-77f5-4010-b2cc-ee2156b969c6)

First-order terms (elements of the sum) quantify the relevance of each input feature to the prediction, and form the explanation. Although simple and straightforward, this method is unstable when applied to deep neural networks.
In LRP, we are propagating the prediction f(x) backward in the neural network, by means of purposely designed local propagation rules.
Let j and k be neurons at two consecutive layers of the neural network. Propagating relevance scores $(R_k )_k$ at a given layer onto neurons of the lower layer is achieved by applying the rule:

![image](https://github.com/user-attachments/assets/70a1d165-43e8-4a22-84fe-29b3844bbcc2)

The quantity $z_{jk}$ models the extent to which neuron j has contributed to make neuron k relevant. The denominator serves to enforce the conservation property. The propagation procedure terminates once the input features have been reached.

![image](https://github.com/user-attachments/assets/1e95b862-5dc5-4e13-989c-3aa699a8358e)


## LoRA
LoRA (Low-Rank Adaptation) is a method for fine-tuning a pre-trained model by modifying a small portion of its parameters. This approach enables the efficient adaptation of large models to specific tasks, and minimizing the computational expense and time needed for fine-tuning.
In LoRA, instead of updating all of the weights of a given weight matrix $W∈R^{d_i×d_o}$, we are only learning a low rank approximation of the of the update ΔW:

AB≈ΔW

Where $A∈R^{d_i×r},B∈R^{r×d_o}$ and $r≪d_i,d_o$ is the rank.

![image](https://github.com/user-attachments/assets/a0bc736e-fb41-45af-ba3f-595a8ddfcf23)

## DoRA
DoRA (Decomposed Low-Rank Adaptation) builds on the LoRA method by leveraging the principle that any vector can be decomposed into its magnitude and direction. In DoRA, instead of fine-tuning the weights directly, they are split into two components:
- Magnitude: the size or scale of the weights
- Direction: the orientation or direction of the weights

This decomposition is applied to the entire weight matrix, with each column representing the connections from all inputs to a specific output neuron. The directional matrix is calculated as $V=\frac{W+AB}{||W+AB||}$, where W is the original weight matrix and AB is the low-rank adaptation. Standard LoRA is then applied to obtain the updated weights as $W_{updated}=m⋅\frac{W+AB}{||W+AB||}$, where m is the scaling factor.

![image](https://github.com/user-attachments/assets/31819291-d92d-4680-9758-02506dde45f9)


## Project goal
The goal of this project is to compare attention-based visualizations of classifications made by a Transformer-based model for vision tasks, both before and after fine-tuning. The objective is to understand how fine-tuning impacts the importance of tokens in the model's classification process.

## Dataset
The dataset we used for our project consists of images from 100 sports categories and was taken from: https://www.kaggle.com/datasets/gpiosenka/sports-classification

## Method
In this work, we compute relevancy for Transformer networks. The method assigns local relevance based on the Deep Taylor Decomposition principle and then propagates these relevancy scores through the layers. <br>
The method employs LRP-based relevance to compute scores for each attention head in each layer of a Transformer model. It then integrates these scores throughout the attention graph, by incorporating both relevancy and gradient information, in a way that iteratively removes the negative contributions. The result is a class-specific visualization for self-attention models. <br>
**Relevance and gradient diffusion:** Let's consider a Transformer model M consisting of B blocks, where each block b is composed of self-attention, skip connections, and additional linear and normalization layers in a certain assembly. The model takes as an input a sequence of s tokens, each of dimension d, with a special token for classification, commonly identified as the token [CLS]. M outputs a classification probability vector y of length C, computed using the classification token. The self-attention module operates on a small sub-space $d_h$ of the embedding dimension d, where h is the number of “heads”, such that $hd_h$ = d. The self-attention module is defined as follows: <br>
![image](https://github.com/user-attachments/assets/abe1a991-35a8-4651-8dcc-5a38b4abbfef) <br>
where (·) denotes matrix multiplication, $O(b) ∈R^{h×s×d_h}$ is the output of the attention module in block b, $Q^b,K^b,V^b∈R^{h×s×d_h}$ are the query key and value inputs in block b, namely, different projections of an input x(n) for a self-attention module. <br>
$A^b∈R^{h×s×s}$ is the attention map of block b, where row i represents the attention coefficients of each token in the input with respect to the token i.<br>
Following the propagation procedure of relevance and gradients, each attention map $A^b$ has its gradients ${∇A}^b$, and relevance $R^{n_b}$ , with respect to a target class t, where $n_b$ is the layer that corresponds to the softmax operation of block b, and $R^{n_b}$ is the layer’s relevance. The final output $C ∈R^{s×s}$ of the method is then defined by the weighted attention relevance:<br>
![image](https://github.com/user-attachments/assets/b68a44b1-effc-4bf4-96fc-912be55459b7)<br>
Where ⊙ is the Hadamard product, and $E_h$ is the mean across the “heads” dimension, and $R^{n_b}$ defined as: <br>
![image](https://github.com/user-attachments/assets/18166adf-a3e0-4790-81a9-3cfb8c62336a)<br>
where we consider only the elements that have a positive weighed relevance (q).

**Obtaining the image relevance map:** The resulting explanation of our method is a matrix C of size s×s, where s represents the sequence length of the input fed to the Transformer. Each row corresponds to a relevance map for each token given the other tokens. Since the work focuses on classification models, only the [CLS] token, which encapsulates the explanation of the classification, is considered. The relevance map is, therefore, derived from the row $C_{[CLS]} ∈R^s$ that corresponds to the [CLS] token. This row contains a score evaluating each token’s influence on the classification token. <br>
In vision models, such as ViT we used for our project, the content tokens represent image patches. To obtain the final relevance map, we reshape the sequence to the patches grid size. for a square image, the patch grid size is $\sqrt{s-1}× \sqrt{s-1}$. This map is upsampled back to the size of the original image using bilinear interpolation.

![image](https://github.com/user-attachments/assets/bacfd6c2-286f-405d-8d0e-6f192a94214b)


## Experiments and results
In this experiment, we investigated the impact of fine-tuning using DoRA on local relevance, comparing it to traditional feature extraction through attention visualization. We employed a pre-trained Vision Transformer (ViT) model, which was initially trained on a diverse range of datasets, including ImageNet, CIFAR-100, and VTAB, encompassing various content types.<br>
We began by applying feature extraction with the pre-trained ViT model to adapt it to our specific dataset, which consists of images from several sports categories, spanning 100 classes. The results from this feature extraction process are detailed below:<br>
The accuracy of the validation set: 97.4%.

![image](https://github.com/user-attachments/assets/8b74e7ae-777a-4744-acb3-689fdf5edbbf)

Next, we applied DoRA to the pre-trained model, supplementing the regular feature extraction process. The results obtained using DoRA are outlined below:<br>
The accuracy of the validation set: 97.2%.

![image](https://github.com/user-attachments/assets/196ca573-28f2-40d9-a359-36bb7f284356)

We observed several noteworthy points:
* Improved class prediction with feature extraction: Our predictions for the relevant class depicted in the image were more accurate when using feature extraction
* Similarity in visual attention with DoRA: The visual attention maps generated with DoRA exhibited similar patterns across different classes (both the actual class and a randomly selected alternative)
* 	Noise in attention maps with feature extraction: When using feature extraction, the visual attention for the incorrect class (not the actual one) included noise, highlighting irrelevant areas of the image
* 	 lightly better validation score with feature extraction: The validation accuracy was marginally higher when using feature extraction compared to DoRA


## Conclusions
Based on our findings, we propose the following conclusions:
The use of DoRA may lead to overfitting compared to feature extraction, as evidenced by the slightly lower validation score and the reduced noise in the attention maps for irrelevant classes. The fact that DoRA involves learning more weights than feature extraction suggests that our dataset might be too small to capture the true attention relevance for each class effectively.
While DoRA highlights meaningful parts of the image, such as people and the ball, these relevant features appear consistently across different classes. In contrast, with feature extraction, the meaningful parts vary between classes, though this also introduces more noise when visualizing irrelevant classes.

## Future work
-	Expand to additional datasets: Extend our study to include a broader range of datasets to validate the generalizability of our findings.
-	Explore alternative fine-tuning methods: Investigate different fine-tuning techniques to further optimize model performance and understand their impact on attention-based visualizations.
-	Deepen understanding of the attention matrix: Conduct a more in-depth analysis of how the attention matrix influences the model's predictions, potentially uncovering new insights into the decision-making process of Transformer-based models.




# How to run
**need to complete**
## Environments settings
1. Clone to a new directory:
  `git clone <URL> <DEST_DIR>`
2. `cd /path/to/DEST_DIR`
3. `pip install -r requirements.txt`
4. Download ImageNet subset from:
  https://www.kaggle.com/datasets/tusonggao/imagenet-validation-dataset/code
5. Move the images directory to:
  `DEST_DIR/../archive/imagenet_validation`

## Execution commands

`python train_evaluate/train_model.py --init {paper_init,svd_init} [ > log.txt]`
* `--init`: determines the LoRA matrices initialization method (default: `paper_init`)
* Recommended: pipe the output to `log.txt` file
* Results will appear in `DEST_DIR/results`

Description:
1. Initiates the `resnet18` model, pretrained on ImageNet
2. Loads the pre-downloaded ImageNet dataset and splits train/val/test subsets
3. Per each compression method (`sparse`, `int1`):
  * Compresses the model
  * Initiates LoRA appended to FC layer(s)
  * Sweeps LoRA rank values, and uses Optuna to find the best training hyper-parameters per each rank
  * Outputs the results to a dedicated directory
3. Results directory contains:
  * `evaluation.csv`: summary of evaluation accuracy for test subset per each LoRA rank
  * `acc_quant=COMP_TYPE_r=RANK.png`: accuracy per epoch (train, validation), for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank)
  * `loss_quant=COMP_TYPE=r_RANK.png`: loss per epoch (train, validation), for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank)
  * `quant=COMP_TYPE_r=RANK_eval_acc=ACCUR.ckpt`: model post-training parameters, for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank), with test accuracy of ACCUR
  * `quant=COMP_TYPE_r=RANK_optimization_history.html`: Optuna trials summary for COMP_TYPE (`sparse`, `int1`) and RANK (LoRA rank) hyper-parameter tuning


# Ethics Statement
?

# Credits
- LRP - https://iphome.hhi.de/samek/pdf/MonXAI19.pdf
- Hila Chefer:
  * Transformer-Explainability - https://github.com/hila-chefer/Transformer-Explainability 
  * Robust-ViT - https://github.com/hila-chefer/RobustViT
- Dataset - https://www.kaggle.com/datasets/gpiosenka/sports-classification
