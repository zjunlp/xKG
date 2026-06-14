\title{
LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies
}

\author{
Jia Shi ${ }^{1}$ Gautam Gare ${ }^{1}$ Jinjin Tian ${ }^{1}$ Siqi Chai ${ }^{1}$ Zhiqiu Lin ${ }^{1}$ Arun Vasudevan ${ }^{1}$ Di Feng ${ }^{23}$ \\ Francesco Ferroni ${ }^{24}$ Shu Kong ${ }^{56}$
}

\begin{abstract}
We tackle the challenge of predicting models' Out-of-Distribution (OOD) performance using indistribution (ID) measurements without requiring OOD data. Existing evaluations with "Effective Robustness", which use ID accuracy as an indicator of OOD accuracy, encounter limitations when models are trained with diverse supervision and distributions, such as class labels (Vision Models, VMs, on ImageNet) and textual descriptions (Visual-Language Models, VLMs, on LAION). VLMs often generalize better to OOD data than VMs despite having similar or lower ID performance. To improve the prediction of models' OOD performance from ID measurements, we introduce the Lowest Common Ancestor (LCA)-on-the-Line framework. This approach revisits the established concept of LCA distance, which measures the hierarchical distance between labels and predictions within a predefined class hierarchy, such as WordNet. We assess 75 models using ImageNet as the ID dataset and five significantly shifted OOD variants, uncovering a strong linear correlation between ID LCA distance and OOD top-1 accuracy. Our method provides a compelling alternative for understanding why VLMs tend to generalize better. Additionally, we propose a technique to construct a taxonomic hierarchy on any dataset using $K$-means clustering, demonstrating that LCA distance is robust to the constructed taxonomic hierarchy. Moreover, we demonstrate that aligning model predictions with class taxonomies, through soft labels or prompt engineering, can enhance model generalization. Open source code in our Project Page.
\end{abstract}

\footnotetext{
${ }^{1}$ Carnegie Mellon University ${ }^{2}$ Work done at Argo AI GmbH ${ }^{3}$ Now at Apple ${ }^{4}$ Now at Nvidia ${ }^{5}$ Texas A\&M University ${ }^{6}$ University of Macau. Correspondence to: Jia Shi <jiashi@alumni.cmu.edu>, Shu Kong <skong@um.edu.mo>.

Proceedings of the $41^{\text {st }}$ International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).
}
![](assets/asset_1.jpg)

Figure 1. Correlation between LCA distance and out-ofdistribution (OOD) performance in Vision and VisionLanguage Models (VLMs). In both panels, the X-axis represents the top-1 accuracy on ObjectNet (OOD test dataset). The Y-axes depict the top-1 accuracy (left-axis) and LCA distance (right-axis) on ImageNet (ID test dataset). The left plot reveals a divergent trend where Vision Models (VMs) show a trade-off between OOD and ID accuracy, while VLMs tend to maintain higher OOD accuracy regardless of ID performance. The right plot demonstrates a unified, strong positive correlation between LCA distance and OOD accuracy for both VMs and VLMs, showing that LCA distance is a robust metric for evaluating model generalization across different architectures, model modalities, and training data sources.

\section*{1. Introduction}

Generalizing models trained on in-distribution (ID) data to out-of-distribution (OOD) conditions is a notoriously difficult task. Distribution shifts undermine the independent and identically distributed (IID) assumption between training and testing data, challenging the model's robustness. Numerous OOD datasets have been proposed to study the effects of different interventions, such as temporal shifts ( Hu et al., 2022; Lomonaco \& Maltoni, 2017; Lin et al., 2021), artificial noise (Hendrycks \& Dietterich, 2019; Arjovsky et al., 2019; Larochelle et al., 2008), and natural distribution shifts (Hendrycks et al., 2021; Hendrycks \& Dietterich, 2019; Barbu et al., 2019; Recht et al., 2019). Maintaining model robustness becomes significantly more difficult with severe visual shifts in the image domain. However, many
![](assets/asset_2.jpg)

Figure 2. Comparison of our setting with prior work. Left: prior work settings such as Accuracy-on-the-line (Miller et al., 2021) and Agreement-on-the-line (Baek et al., 2022). Right: our setting. To the best of our knowledge, LCA-on-the-line is the first approach to uniformly measure model robustness across VMs and VLMs on OOD datasets with significant distribution shifts (ImageNet-S/R/A/O).
studies evaluate generalization on OOD datasets with limited visual shifts or only involve artificial noise, such as ImageNet-v2 or ImageNet-C (Recht et al., 2019; Arjovsky et al., 2019). Such datasets fail to fully reflect a model's generalization capability when confronted with severe distribution shifts (Hendrycks et al., 2021; Hendrycks \& Dietterich, 2019; Barbu et al., 2019), as there is often limited transfer of robustness from synthetic to natural distribution shifts (Taori et al., 2020).

In the realm of model generalization, numerous attempts have been made to predict a model's performance on OOD datasets based on in-distribution measurements, following the concept of effective robustness (Taori et al., 2020). These approaches, referred to as 'X-on-the-line' (Miller et al., 2021; Baek et al., 2022), suggest that a model's OOD performance is correlated to in-distribution accuracy (Miller et al., 2021; Recht et al., 2019; Miller et al., 2020; Roelofs et al., 2019) or models consensus on in-distribution accuracy (Jiang et al., 2021; Baek et al., 2022).

Moreover, several prior attempts rely on domain generalization strategies that necessitate prior knowledge of the target domain or require an estimation of OOD domain information (Chen et al., 2021; Li et al., 2022a). These can lead to computationally intensive processes, particularly when involving multiple models or inferences (Baek et al., 2022; Deng et al., 2022).

Most prior research has focused solely on estimating generalization among vision models (VMs) supervised on class labels trained on ImageNet (Taori et al., 2020; Mustafa et al., 2020). Emerging large-scale Vision-Language Models (VLMs) trained on datasets like LAION demonstrate exceptional generalization performance on out-of-distribution (OOD) data. However, as shown on the left plot of Fig. 1, existing evaluation (Miller et al., 2021) using ID accuracy fail to explain the effective robustness (Taori et al., 2020) gap
between VMs and VLMs. This underscores the necessity to evaluate and compare models across different families under a unified evaluation framework. Recently, (Shi et al., 2023) observed the same problem and proposed evaluating OOD accuracy using multiple ID test sets, but their method requires multiple evaluation runs.

Unlike VMs, VLMs leverage more diverse training data, contrastive loss, and language supervision. There have been attempts to measure VLM generalization (HaoChen et al., 2021; Fang et al., 2022; Schuhmann et al., 2022; Kaur et al., 2022), specifically suggesting that diversity in training data is an indicator of model generalization. However, it is nontrivial to measure data diversity, and even collect and train on such large-scale diverse data (Schuhmann et al., 2022).

Prior attempts lack a unified, simple measurement for both VMs and VLMs to explain model generalization and convert it into actionable improvements. To address the issues of (1) lack of unified metrics for VLMs and VMs, or models trained on different data sources; (2) need for robustness to large domain shifts; (3) desire for computationally efficient metrics, we propose adopting the Lowest Common Ancestor (LCA) distance to measure model generalization. The LCA distance is the taxonomic distance between labels and predictions, given a predefined class hierarchy, such as WordNet. Through a series of empirical experiments involving 75 models ( 36 VMs and 39 VLMs) (cf. Fig. 2), we show that the in-distribution LCA distance strongly correlates with multiple ImageNet-OOD datasets under severe visual shifts (cf. Fig. 1 right plot). This finding may help explain the surprising result that zero-shot vision-language models with poor top-1 accuracy generalize better to novel datasets compared to state-of-the-art vision models. This spurs us to further investigate and discuss the potential of the LCA benchmark for improving model generalization. We also discuss the suitability of LCA as a generalization indicator in Section 3.

In summary, we make the following major contributions: (1) We propose the Lowest Common Ancestor (LCA) distance as a new metric for evaluating model generalization. This benchmark utilizes class hierarchies, such as WordNet, which encode relationships between classes. (2) We validate our benchmarking strategy through large-scale experiments, analyzing 75 models across five ImageNet-OOD datasets. Our findings reveal a strong linear correlation between in-distribution LCA and OOD Top-1 performance, thus establishing the 'LCA-on-the-Line' framework. (3) We offer a thorough analysis of the connection between LCA and model generalization, providing new insights to inspire further research in this area. (4) For datasets without a predefined hierarchy, we introduce a method for constructing latent hierarchies using K-means clustering. Our results demonstrate that the LCA distance is robust to variations in underlying taxonomies or hierarchies. (5) We illustrate the potential of this benchmark by demonstrating how model generalization can be enhanced by aligning model predictions with the class hierarchy.

\section*{2. LCA Distance Measures Misprediction Severity}

We propose using the in-distribution Lowest Common Ancestor (LCA) distance, also known as taxonomy loss, as a predictor for model generalization. Here, we formally define how taxonomy loss can be measured using in-distribution data. Taxonomy loss measures the class ranking difference between a model's prediction based on class likelihood, and a predefined class order encoded by class taxonomy. Lower taxonomy loss is expected when a model assigns higher likelihood to classes that are semantically closer to the ground-truth class, in other words, 'making better mistakes' (Bertinetto et al., 2020; Peri et al., 2023). For example, if a cat image is predicted as a dog by model-A and as a car by model-B, model-A would have a lower LCA distance as it makes a better mistake than model-B. Following previous research (Bertinetto et al., 2020; Deng et al., 2009b), we use WordNet (Miller et al., 1990), a large-scale lexical database inspired by psycholinguistic theories of human lexical memory (Miller, 1995), to encode class taxonomy. The WordNet taxonomy is well suited for the widely used ImageNet dataset which builds on WordNet. An example of LCA distance is shown in Fig 3.
Given two classes, $y$ (the ground-truth class) and $y^{\prime}$ (the prediction class), we define the LCA distance according to (Bertinetto et al., 2020) as
\[
D_{L C A}\left(y^{\prime}, y\right):=f(y)-f\left(N_{L C A}\left(y, y^{\prime}\right)\right)
\]
where $f(y) \geq f\left(N_{L C A}\left(y, y^{\prime}\right)\right)$ and $N_{L C A}\left(y^{\prime}, y\right)$ denotes the lowest common ancestor class node for classes $y$ and $y^{\prime}$ within the hierarchy, and $f(\cdot)$ represents a function of a node,

Taxonomy distance as a measurement of semantic severity of mistake
![](assets/asset_3.jpg)

Figure 3. LCA distance visualization. Our method estimates a model's generalization based on its in-distribution semantic severity of mistakes. We use the 'Lowest Common Ancestor' (LCA) distance to rank the distance between the model's prediction and the ground-truth class within a predefined taxonomic hierarchy, such as WordNet. The LCA distance is proportional to the shortest path from the prediction to the ground-truth class in the hierarchy.
such as the tree depth or entropy. We use the information content as described in (Valmadre, 2022). For each sample $X_{i}$ in the given dataset $\mathcal{M}:=X_{1}, \ldots, X_{n}$ :
$D_{L C A}(\operatorname{model}, \mathcal{M}):=\frac{1}{n} \sum_{i=1}^{n} D_{L C A}\left(\widehat{y}_{i}, y_{i}\right) \Longleftrightarrow y_{i} \neq \widehat{y}_{i}$
where $\widehat{y}_{i}$ is the predicted class for sample $X_{i}$ using the model, $y_{i}$ is the ground-truth class for sample $X_{i}$, and $y_{i} \neq \widehat{y}_{i}$. Intuitively, a model with a lower LCA distance demonstrates a greater semantic understanding of class ontology in WordNet.
We can also derive the generalized form of LCA distance to settings where the model outputs a distribution over all possible classes for each sample (like using softmax), please refer to appendix D. 3 for details.

\section*{3. Discussion: The Suitability of LCA as a Benchmark for Model Generalization}

This section explores the hypothesis linking LCA distance with a model's generalization ability and discusses how these insights can be meaningfully and actionably applied.
Our primary motivation is to use class hierarchy to capture correlation invariances across training environments, as proposed in the seminal work on 'invariant risk minimization' (Arjovsky et al., 2019). Since the class hierarchy remains consistent across both ID and OOD datasets, it can serve as a surrogate measure of the model's invariant features. Models that generalize well to OOD datasets typically learn universal or non-spurious features from the training dataset that are transferable to OOD datasets (Makar et al.,
![](assets/asset_4.jpg)

Figure 4. Capturing transferable features for model generalization. ImageNet-R maintains shape information (Geirhos et al., 2018) like 'long neck', 'big belly', and 'long legs'. We hypothesize that models with good generalization should capture these transferable features rather than succumbing to spurious correlations such as 'grass', thereby tending to predict classes that are semantically closer to the ground-truth. Such models are expected to have low LCA distances between their predictions and the ground-truth.
2022). Such models are more likely to misclassify an ostrich as another bird rather than a lion. These taxonomybased mispredictions, quantified using the LCA distance, are shown to be a better indicator of a model's OOD performance in this work.

Obstacles to Model Generalization. In deep learning, models often learn predictive features from images by creating discriminative associations to class labels. This approach is susceptible to spurious correlations in the training data (Sturm, 2014; Torralba \& Efros, 2011; Jabri et al., 2016). For instance, a model might erroneously associate the class 'ostriches' with the feature 'green grass' in the background, as ostriches often appear in grasslands. These correlations may fail when applied to an OOD dataset that only depicts the semantic concept of 'ostriches' (Zhang et al., 2021).
Essentials for Model Generalization. ImageNet-R is a severely shifted OOD dataset where, despite significant distribution shifts, humans can effortlessly identify the correct classes. This is because humans can discern stable features across environments. A model's generalization capability depends on the transferability of the associations learned during training. As benchmarks often simulate humanworld ontology, ideally, only features that align with human understanding of object semantics are universally transferable to any constructed OOD dataset. This underscores the importance of identifying transferable features aligning ontology that contribute to robust model generalization.
How can we measure what features a model has learned as predictive during training? The decision-making process of deep neural networks trained end-to-end has become less interpretable. While there have been attempts to decipher this process by forming decision-tree-like models (Wan
et al., 2020; Gare et al., 2022) or through learnable activation functions (Liu et al., 2024), these efforts have not linked this understanding to measure model generalization.
Class Taxonomy Alignment as a Feature Measurement. Class taxonomy or ontology has been widely utilized in literature to indicate class formation (Deng et al., 2009b; Van Horn et al., 2018) and semantic relationships between classes (Frome et al., 2013; Barz \& Denzler, 2019; Wan et al., 2020; Redmon \& Farhadi, 2017; Lin et al., 2022), offering a hierarchical organization of classes or categories.

As WordNet encodes class ontology, we hypothesize that transferable features are more likely to be shared among neighboring classes in the hierarchy (e.g., ostrich and crane). In contrast, confounding features are less supported by the hierarchy and tend to appear in less relevant classes that are often more distant in the hierarchy (e.g., lion and ostrich). When a model makes a mistake, its secondary prediction class can provide insight into the predictive features the model has learned during training. Specifically, it reflects that the model perceives the label class and the secondary prediction class to be more similar to each other based on these predictive features.

Consequently, a model that captures more transferable features tends to 'make better mistakes (Bertinetto et al., 2020)' by predicting classes that are semantically closer to the ground-truth class. As illustrated in Fig. 4, models that learns to associate ostriches with features like 'long legs' and 'long neck', which are more transferable to OOD datasets, will likely predict classes like flamingos or cranes. In contrast, a model influenced by spurious correlations and associating ostriches with grass might predict a semantically distant class, like jaguars or lions, which also often appear on grass.

Our method involves measuring model generalization based on the semantic severity of mistakes on in-distribution data. We use the LCA distance, the taxonomic distance between the model's prediction and the ground-truth class in a predefined taxonomic hierarchy like WordNet. If a model consistently makes better mistakes on in-distribution data, we can reasonably assume that the model has captured more transferable features for class discrimination.

Class Taxonomy and Mistake Severity. The severity of a mistake in many studies is quantified as the shortest path from the prediction node to the lowest common ancestor (LCA) node in a predefined class hierarchy. This metric, known as 'LCA distance' or 'hierarchical error', was used in the early years of the ImageNet challenge (Deng et al., 2009b). However, it was largely dismissed as it was widely believed to follow the same ordering as Top 1 accuracy (Bertinetto et al., 2020). We revisit this metric and empirically demonstrate that Top 1 accuracy and LCA distance do
not always align when VLMs are involved, challenging the common notion. We also appeal for community's attention to revisit this metric with its potential usage in measuring a model's feature awareness to indicate generalization.
Causal/Invariant Representation Learning for OOD Generalization. Recently, there has been an increase in OOD generalization research towards formulating training and testing distributions with causal structures (Arjovsky et al., 2019; Bühlmann, 2020; Peters et al., 2016), where distribution shifts primarily arise from interventions or confounding factors. Building upon this, methods (Schölkopf et al., 2021; Shen et al., 2022; Subramanian et al., 2022) such as CausalVAE (Yang et al., 2021) have been proposed, leveraging learned causal representations to capture the causal relationships underlying the data generation process (Kaur et al., 2022), which helps mitigate the distributional shifts caused by interventions.
While the connection between OOD generalization and causal concepts is not entirely novel, previous attempts have focused on the causal structure at the latent or abstract level, lacking both interpretability and transparency. Our method aligns with this growing interest in causal/invariant learning, which aims to capture the invariant latent data generation process (Kaur et al., 2022). One should expect a model prediction that better aligns with the data generation process to be more robust under intervention, thus generalizing better. Although it is less feasible to model the data generation process of natural images (ImageNet), we essentially follow the same intuition and hypothesize that the WordNet class hierarchy serves as an approximation of invariant correlations between class concepts across environments (Arjovsky et al., 2019; Santurkar et al., 2020), robust to spurious relations in images or shortcuts in learning (Makar et al., 2022). WordNet is a widely recognized and effective means of encoding semantic relationships between concepts, making it an appropriate proxy for aligning human semantic knowledge (Miller et al., 1990). Unlike previous work, WordNet hierarchy provides interpretability, adding a level of transparency to our understanding of model generalization.
LCA Illustration with Simulated Data. To illustrate our hypothesis that LCA distance can identify features supported by hierarchy, we created a controlled example using a simulated dataset, detailed in Appendix C. In this example, the data generation process is fully controlled. We designed a feature space that includes: 1) transferable causal features supported by hierarchy, 2) non-transferable confounding features not supported by hierarchy, and 3) random noise. Two logistic regression models were trained to mimic models capturing different predictive variables from the training data: one relying on the causal features and the other on the confounding features. The simulation results indicated
that the model using causal features supported by hierarchy, which exhibited lower LCA distance, had better out-ofdistribution (OOD) accuracy on the in-distribution (ID) test set, despite the model using confounding features achieving better ID accuracy. This example suggests that LCA can effectively identify models that capture relationships aligned with the hierarchical structure. Details in code snippet.

\section*{4. Experiments}

We present experiments benchmarking the relationship between Lowest Common Ancestor (LCA) and generalization.

Dataset Setup. We leverage 75 pretrained models sourced from open repositories on GitHub for empirical analysis. Our selection comprises 36 Vision Models (VMs) pretrained on ImageNet and supervised from class labels, alongside 39 Vision-Language Models (VLMs) that incorporate language as part of the supervision. A comprehensive list of model details, ensuring reproducibility, is provided in Appendix A. We use ImageNet (Deng et al., 2009b) as the source indistribution (ID) dataset, while ImageNet-v2 (Recht et al., 2019), ImageNet-Sketch (Hendrycks \& Dietterich, 2019), ImageNet-Rendition (Hendrycks et al., 2021), ImageNetAdversarial (Hendrycks et al., 2021), and ObjectNets (Barbu et al., 2019) are employed as out-of-distribution datasets, exemplifying severe natural distribution shifts. The ImageNet hierarchy, as depicted in (Bertinetto et al., 2020), is utilized.

Although ImageNet-v2 is predominantly deemed an OOD dataset in most prior literature (Shankar et al., 2020; Miller et al., 2021; Baek et al., 2022), our experiments suggest that ImageNet-v2 aligns more closely with ImageNet than other OOD datasets; we delve into these details in Appendix B.
Note that the terms in-distribution (ID) and out-ofdistribution (OOD) are not model-specific in this context. Due to the varying distribution of training data across different models, ImageNet may not necessarily represent ID data for models like CLIP, where the training data distribution is not explicitly known. Instead, ID and OOD are relative concepts. ImageNet is used as a reference anchor dataset, serving as a baseline to evaluate the generalization capabilities of models on OOD datasets. This approach aligns with prior work, allowing us to consistently measure the shift in performance from ID to OOD datasets, despite the differences in the training data distributions of the models.

Metric Setup. For our correlation experiment, we use $R^{2}$ (Coefficient of Determination) and PEA (Pearson correlation coefficient) to measure the strength and direction of linear relationships between two variables. Additionally, we employ KEN (Kendall rank correlation coefficient) and SPE (Spearman rank-order correlation coefficient) to assess the correspondence of the rankings of two variables.
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline Model & \multicolumn{2}{|c|}{ImgN} & ImgN-v2 & ImgN-S & ImgN-R & ImgN-A & ObjNet \\
\hline & LCA $\downarrow$ & Top1 $\uparrow$ & Topl $\uparrow$ & Top $1 \uparrow$ & Top1 $\uparrow$ & Top 1 $\uparrow$ & Top1 $\uparrow$ \\
\hline ResNet18 & 6.643 & 0.698 & 0.573 & 0.202 & 0.330 & 0.011 & 0.272 \\
\hline ResNet50 & 6.539 & 0.733 & 0.610 & 0.235 & 0.361 & 0.018 & 0.316 \\
\hline CLIP_RN50 & 6.327 & 0.579 & 0.511 & 0.332 & 0.562 & 0.218 & 0.398 \\
\hline CLIP_RN50x4 & 6.166 & 0.641 & 0.573 & 0.415 & 0.681 & 0.384 & 0.504 \\
\hline
\end{tabular}

Table 1. Model performance corresponds to mistake severity. Results are measured by LCA $\downarrow$ and Top $1 \uparrow$, respectively. indicate measurements on a given dataset. We present model comparisons across VMs and VLMs families. In-distribution LCA distance indicate severely shifted OOD performance (ImageNet-S/R/A/O) better than in-distribution (ImageNet) Top1 accuracy (except for ImageNet-v2). Full 75 models evaluation in Table 2.

The importance of these measurements lies in their different focuses. Linearity measures, such as $R^{2}$ and PEA, are primarily concerned with the fit of a linear model to data points, allowing us to quantify the predictability of changes in one variable based on the other. Ranking measures, like KEN and SPE, provide insights into how the rankings of variables relate to each other, which is crucial in downstream applications such as image retrievals and search engine optimization, where understanding and predicting the ordering of data points is often more important than predicting their exact values. For prediction experiments, we utilize MAE (Mean Absolute Error) to quantify the absolute difference between predictions and ground-truth.

\subsection*{4.1. LCA-on-the-Line: In-Distribution Taxonomic Distance (LCA) as an Out-of-Distribution (OOD) Performance Predictor}

Accuracy-on-the-line (Miller et al., 2021) corroborated that a model's in-distribution (ID) accuracy and its out-of-distribution (OOD) accuracy are largely considered to be strongly correlated. This potent correlation forms a significant baseline for comparison in our research. Unlike the framework presented in (Miller et al., 2021), which only compares models within the same modality, our work bridges the gap by contrasting models of different modalities, involving both Vision Models (VM) and VisionLanguage Models (VLM). In addition to the Top1 OOD accuracy, we also incorporate Top5 OOD accuracy, yielding a more comprehensive evaluation of model generalization.

As displayed in Table 1 and 2, the ImageNet in-distribution accuracy (Miller et al., 2021) forms a robust predictor for most OOD datasets, when the comparison is limited to models with similar setups (VMs or VLMs). However, this predictor fails to provide a unified explanation of generalization across models from both families. As highlighted in Figure 5 (indicated in red line), when adhering to Accuracy-on-the-Line’ (Miller et al., 2021), all four OOD datasets plotted showcase two separate linear trends, representing models that belong to each family. This observation aligns with (Cherti et al., 2022), where it was found that VLM models, despite exhibiting significantly lower ID accuracy,
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline \multicolumn{2}{|l|}{Element} & \multicolumn{2}{|l|}{ImgN-v2} & \multicolumn{2}{|r|}{ImgN-S} & \multicolumn{2}{|l|}{ImgN-R} & \multicolumn{2}{|l|}{ImgN-A} & \multicolumn{2}{|r|}{ObjNet} \\
\hline ID & OOD & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA \\
\hline Top1 & Top1 & 0.962 & 0.980 & 0.075 & 0.275 & 0.020 & 0.140 & 0.009 & 0.094 & 0.273 & 0.522 \\
\hline LCA & Top1 & 0.339 & 0.582 & 0.816 & 0.903 & 0.779 & 0.883 & 0.704 & 0.839 & 0.915 & 0.956 \\
\hline Topl & Top5 & 0.889 & 0.943 & 0.052 & 0.229 & 0.004 & 0.060 & 0.013 & 0.115 & 0.262 & 0.512 \\
\hline LCA & Top5 & 0.445 & 0.667 & 0.811 & 0.901 & 0.738 & 0.859 & 0.799 & 0.894 & 0.924 & 0.961 \\
\hline
\end{tabular}

Table 2. Correlation measurement by $R^{2}$ and $P E A$ of ID LCA/Top1 with OOD Top1/Top5 across 75 models ( 36 VMs and 39 VLMs) as shown in Figure 5. We demonstrate that LCA has a strong correlation with OOD performance on all listed datasets (except ImageNet-v2). We take the absolute value of all correlations for simplicity. Full table containing results of VMs-only and VLMs-only in Table 11. Measurements from the KEN and SPE show a similar trend as seen in Section F.
\begin{tabular}{llllll}
\hline Methods & ImgN-v2 & ImgN-S & ImgN-R & ImgN-A & ObjNet \\
\hline ID Top1 (Miller et al., 2021) & $\mathbf{0 . 0 4 0}$ & 0.230 & 0.277 & 0.192 & 0.178 \\
AC (Hendrycks \& Gimpel, 2017) & $\underline{0.043}$ & $\underline{0.124}$ & $\mathbf{0 . 1 1 3}$ & 0.324 & $\underline{0.127}$ \\
Aline-D (Baek et al., 2022) & 0.121 & 0.270 & 0.167 & 0.409 & 0.265 \\
Aline-S (Baek et al., 2022) & 0.072 & 0.143 & 0.201 & $\underline{0.165}$ & 0.131 \\
(Ours) ID LCA & 0.162 & $\mathbf{0 . 0 9 3}$ & $\underline{0.114}$ & $\mathbf{0 . 1 0 3}$ & $\mathbf{0 . 0 4 8}$ \\
\hline
\end{tabular}

Table 3. Error prediction of OOD datasets across 75 models of diverse settings measured by MAE loss $\downarrow$. We mark the best and second best method bold and underline, respectively. Despite ImageNet (ID) accuracy remaining a significant indicator of ImageNet-v2 (OOD) accuracy, the ID LCA serves as a more robust error predictor across the four diverse OOD datasets. Refer to Table 12 for detailed results of VMs-only and VLMs-only.
could attain higher OOD performance than their state-of-the-art VM counterparts.

As shown in Figure 1, our method, adopting in-distribution LCA distance, could unify models from both families. As demonstrated in Table 2 and Figure 5 (colored in green line), the severity of in-distribution mistakes serves as a more effective indicator of model performance than in-distribution accuracy. It consistently exhibits a strong linear correlation with all OOD benchmark accuracies for natural distribution shifts (both $R^{2}$ and the Pearson correlation coefficient exceed 0.7 , while (Miller et al., 2021) drop to 0 in ImageNetA). Notably, our experiments showed that (Miller et al., 2021) is a more reliable indicator solely for ImageNet-v2, given its visual similarity to ImageNet. We will further discuss this in Appendix B.

Our method restores the "on-the-line" linear relationship in front of both VMs and VLMs. Our method provides a compelling alternative to understand why vision-language models with lower in-distribution accuracy might generalize better to OOD datasets than vision models.

\subsection*{4.2. Predicting OOD Performance via ID LCA}

We further highlight the effectiveness of the LCA-on-theLine by estimating model OOD performance using a linear function derived from in-distribution LCA distance. For comparison, we included four competitive baselines: $A v$ erage Confidence (AC), which leverages OOD logits after temperature scaling; two methods from Agreement-on-theLine (Aline-D and Aline-S), utilizing consensus of pairs of
![](assets/asset_5.jpg)

Figure 5. Correlating OOD Top-1/Top-5 accuracy (VM+VLM, 75 models) on 4 ImageNet-OOD datasets visualizing Table 2. The plots clearly demonstrate that the in-distribution LCA distance has a stronger correlation with the model's OOD performance across all OOD datasets than accuracy-on-the-line (Miller et al., 2021). Each plot's x-axis represents the OOD dataset metric (with OOD Top-1 in the top row, and OOD Top-5 accuracy in the bottom row) and y-axis represents ImageNet ID test Top-1 accuracy (left) and LCA (right); Red line (Pink dots: VMs and Red dots: VLMs) represents in-distribution classification accuracy (Top-1); Green line (Green dots: VMs and Blue dots: VLMs) denotes in-distribution taxonomic distance (LCA). As interpreted in Figure 1, accuracy-on-the-line only explains generalization of models within similar settings (VMs or VLMs), but does not unify both settings.
models on OOD benchmarks; and 'Accuracy on the Line' (ID Topl), employing in-distribution accuracy of established measurement models to fit a linear function. Instead of performing a probit transform as done in (Baek et al., 2022) and (Miller et al., 2021), we implemented min-max scaling because LCA does not fall within the [0,1] range.
As illustrated in Table 3, in-distribution LCA distance proves to be a significantly more robust OOD error predictor than other baselines across four OOD benchmarks with varying distribution shifts. This robustness is especially evident for ImageNet-A, an adversarial dataset derived from ResNet50's misclassifications on ImageNet. Consequently, models pre-trained on ImageNet tend to underperform on this dataset, especially those with lower accuracy than ResNet50. This leads to decreased robustness for indistribution indicators like in-distribution accuracy (Miller et al., 2021), methods calibrated from in-distribution validation sets (Hendrycks \& Gimpel, 2017), and OOD agreement of models from different families (Baek et al., 2022). In contrast, LCA, which relies solely on the relative ranking of class predictions from a single model, is less sensitive to these issues and thus delivers more consistent performance. This further underscores the efficacy of LCA as a powerful predictor in challenging OOD scenarios.

\subsection*{4.3. Enhancing Generalization via Taxonomy Alignment}

Building upon the earlier discussion, we explore how the devised method can be utilized to enhance a model's generalization capability.

\subsection*{4.3.1. Inferring Class Taxonomy from a Pretrained Model via K-Means Clustering}

In the previous experiment, we adopted the WordNet hierarchy as class taxonomy to calculate LCA distance. While the number of publicly available datasets providing class taxonomy is limited (Deng et al., 2009b; Van Horn et al., 2018), the usefulness of our method is unquestionable. Hence, we propose a method to construct a latent class taxonomy given a well-trained model on the task, expanding the potential applications of our work. We show that such a constructed taxonomy could achieve similar correlational performance to the WordNet hierarchy.
The essence of class taxonomy lies in its representation of inter-class distance, encoding class proximity, and identifying which classes cluster closely in feature space. In this spirit, we can construct a class taxonomy matrix using K-means clustering on image features. As illustrated in

Latent hierarchy from K-mean Clustering
![](assets/asset_6.jpg)

Figure 6. Hierarchical structure of image feature clustering using K-means. We construct latent hierarchy through K-means clustering on image features extracted from a pre-trained model. $\mathrm{K}=1$ represent the most generalized cluster, then we incrementally increase the granularity by splitting into $\mathrm{K}=2$ and $\mathrm{K}=4$ clusters. Each node in the hierarchy represents a cluster with the number indicating the class indexes assigned to that cluster. Table 4 show that robust performance can be achieved among 75 latent hierarchy constructed from different pretrained models using clustering.
\begin{tabular}{cccccccc}
\hline & \multicolumn{2}{c}{ Element } & ImgN-v2 & ImgN-S & ImgN-R & ImgN-A & ObjNet \\
\cline { 2 - 7 } & ID & OOD & & & \\
\hline Baseline & Top1 & Top1 & $\mathbf{0 . 9 8 0}$ & 0.275 & 0.140 & 0.094 & 0.522 \\
WordNet & LCA & Top1 & 0.582 & $\mathbf{0 . 9 0 3}$ & $\mathbf{0 . 8 8 3}$ & $\mathbf{0 . 8 3 9}$ & $\mathbf{0 . 9 5 6}$ \\
\hline \multicolumn{7}{l}{ LCA (Statistical Measurements calculated from 75} & different Latent Hierarchies) \\
\hline Mean & LCA & Top1 & 0.815 & $\mathbf{0 . 7 7 3}$ & $\mathbf{0 . 7 1 2}$ & $\mathbf{0 . 6 6 2}$ & $\mathbf{0 . 9 3 0}$ \\
Min & LCA & Top1 & 0.721 & 0.715 & 0.646 & 0.577 & 0.890 \\
Max & LCA & Top1 & 0.863 & 0.829 & 0.780 & 0.717 & 0.952 \\
Std & LCA & Top1 & 0.028 & 0.022 & 0.027 & 0.025 & 0.010 \\
\hline
\end{tabular}

Table 4. Correlation measurement ( $P E A$ ) between ID LCA/Top1 and OOD Top1 across 75 latent hierarchies derived from K-means. Our latent hierarchy construction is robust across 75 different source pretrained models: For each source model, we extracted average class features and applied K-means clustering to construct a latent hierarchy. We then calculated the LCA distance based on each hierarchy, and aggregated the statistical metric of the 75 groups' Pearson correlation coefficient ( $P E A$ ) to OOD performance (essentially 75 groups of data from Table 2). We observe that LCA reliably tracks OOD performance even when using different class taxonomies.

Fig. 6, for the ImageNet dataset, we adopt a well-trained model as the source pretrained model and extract average class features to cluster data hierarchically at different levels (we use $\mathrm{n}=9$ for the 1000 -class ImageNet dataset, as $2^{9}<1000$ ), with an increasing number of clusters to indicate class adjacency. K-mean is performed on each level of hierarchy independently. Experiments in Table 4 show that our method is very robust regardless of which model was used as the source model to construct the class hierarchy. This result demonstrate the potential in practice to use a latent hierarchy constructed by only one well-trained model for evaluating all models on a given task. Further implementation details are provided in Appendix E.1.

\subsection*{4.3.2. Using Class Taxonomy as Soft Labels}

In this section, we investigate how leveraging LCA distance can enhance model generalization through improved supervision. Traditional models maximize the likelihood of the top- 1 ground-truth class but often fail to generalize due to overfitting from spurious correlations. We argue that a generalizable model should accurately assign likelihoods to all classes in alignment with the class ontology. Building on this insight, we augment the standard cross-entropy loss, which maximizes the top-1 likelihood, with an auxiliary loss that uses soft labels encoded by the normalized pairwise class distance (LCA distance). This approach treats the problem as multi-label classification (Lin et al., 2022), guiding the model's decision boundary towards a more regularized feature distribution, thereby reducing susceptibility to spurious correlations and improving generalization. We balance the contributions of the cross-entropy and auxiliary losses using a lambda term: $\mathrm{L}=\lambda \mathrm{L}(\mathrm{CE})+L\left(\right.$ soft $\left._{l c a}\right)$. The detailed formulation is provided in Appendix E.2.

WordNet as Soft Labels. To evaluate our approach, we trained linear probe layers on five different models using either cross-entropy loss only (Baseline) or our cross-entropy plus LCA soft loss. We compared their performance on six ImageNet test sets. Inspired by the notion that models exhibit higher confidence where they excel (Wortsman et al., 2022), we applied linear interpolation between layers trained with cross-entropy and our proposed loss as our final classifier $W_{\text {interp }}=\alpha W_{c e}+(1-\alpha) W_{c e+\text { soft }}$. Table 5 shows that incorporating LCA soft loss consistently improved OOD performance without compromising ID performance, indicating more regularized decision boundaries beyond training data knowledge. Ablation study is presented in Table 9.

Latent Hierarchy as Soft Labels. To demonstrate that our method generalizes beyond WordNet hierarchy, we constructed latent hierarchies using K-means clustering on pretrained models, forming soft labels to guide linear probing. We followed the same training procedure as above, using latent hierarchies instead of WordNet to construct soft labels. As shown in Table 6, adopting constructed hierarchies similarly boosted model generalization across all OOD datasets.
VLMs Construct Better Soft Labels Compared to VMs. Drawing on the intuition of model distillation (Hinton et al., 2015), the hierarchy constructed from a model's pretrained features partially encapsulates the model's interpretation of interclass relationships. Thus we also examined if the source model affects the quality of derived soft labels. Figure 7 visualizes pair-wise LCA distance matrices for ImageNet data using hierarchies from different models.

We further conducted a correlation study using latent hierarchies generated from all 75 pretrained models, comparing the source model's ID LCA evaluated on WordNet, with
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline Hierarchy Source: WordNet & \multicolumn{2}{|r|}{ImgNet} & \multicolumn{2}{|r|}{ImgNet-V2} & \multicolumn{2}{|r|}{ImgNet-S} & \multicolumn{2}{|r|}{ImgNet-R} & \multicolumn{2}{|r|}{ImgNet-A} & \multicolumn{2}{|r|}{ObjectNet} \\
\hline Backbone Models & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours \\
\hline ResNet 18 (He et al., 2016) & 69.4 & 69.4 (+0.0) & 56.4 & 56.9 (+0.5) & 19.7 & 20.7 (+1.0) & 31.9 & 33.8 (+1.8) & 1.1 & 1.2 (+0.1) & 27.0 & 28.0 (+1.0) \\
\hline ResNet 50 (He et al., 2016) & 79.5 & 79.8 (+0.3) & 67.9 & 68.6 (+0.7) & 25.5 & 27.7 (+2.2) & 36.5 & 42.5 (+6.0) & 10.3 & 16.2 (+5.9) & 43.2 & 45.5 (+2.3) \\
\hline VIT-B (Dosovitskiy et al., 2020) & 75.8 & 75.9 (+0.1) & 62.9 & 62.8 (-0.1) & 27.0 & 27.6 (+0.6) & 40.5 & 41.5 (+1.0) & 8.0 & 8.6 (+0.6) & 27.6 & 28.1 (+0.5) \\
\hline VIT-L (Dosovitskiy et al., 2020) & 76.8 & 76.8 (+0.0) & 63.9 & 63.8 (-0.1) & 28.4 & 29.2 (+0.8) & 42.2 & 43.6 (+1.4) & 10.6 & 11.5 (+0.9) & 28.7 & 29.0 (+0.3) \\
\hline ConvNext (Liu et al., 2022) & 82.0 & 82.1 (+0.1) & 70.6 & 71.0 (+0.4) & 28.7 & 30.0 (+1.3) & 42.4 & 44.3 (+1.9) & 21.8 & 25.3 (+3.5) & 44.4 & 45.5 (+1.1) \\
\hline Swin Transformer (Liu et al., 2021) & 83.1 & 83.2 (+0.1) & 72.0 & 71.9 (-0.1) & 30.3 & $31.4(+1.1)$ & 43.5 & 45.3 (+1.8) & 29.5 & 32.7 (+3.2) & 48.3 & 49.5 (+1.2) \\
\hline
\end{tabular}

Table 5. Soft labeling with WordNet for Linear Probing. Baseline: Trained with Cross Entropy only; Ours: Trained with Cross Entropy + LCA soft loss + weight linear interpolation of (CE, CE + soft loss) (Wortsman et al., 2022). Results show that integrating soft loss consistently improves model OOD performance, without compromising ID accuracy. Note that in Table 9 of ablation study in pro-OOD setting, we demonstrate that it's possible to further enhance OOD performance at the cost of a slight ID accuracy drop.
\begin{tabular}{cccccccccc}
\hline Backcbone Model:ResNet-18 & \multicolumn{3}{c}{ ImgNet-S } & \multicolumn{3}{c}{ ImgNet-R } & \multicolumn{3}{c}{ ImgNet-A } \\
\hline Hierarchy Sources & Baseline & Interp & Baseline & Interp & Baseline & Interp & \multicolumn{2}{c}{ Objecline } & Interp \\
\hline MnasNet & 19.7 & $20.2(+0.5)$ & 31.9 & $32.4(+0.5)$ & 1.1 & $1.7(+0.6)$ & 27.0 & $28.1(+1.1)$ \\
ResNet 18 & 19.7 & $20.2(+0.5)$ & 31.9 & $32.4(+0.5)$ & 1.1 & $1.8(+0.7)$ & 27.0 & $28.2(+1.2)$ \\
vit-1-14 & 19.7 & $20.8(+1.2)$ & 31.9 & $33.2(+1.3)$ & 1.1 & $2.0(+0.9)$ & 27.0 & $28.3(+1.3)$ \\
OpenCIP(vit-1-14) & 19.7 & $20.9+1.3)$ & 31.9 & $33.7(+1.8)$ & 1.1 & $2.1+1.0)$ & 27.0 & $28.5(+1.5)$ \\
\hline WordNet & 19.7 & $\mathbf{2 1 . 2}(+1.5)$ & 31.9 & $\mathbf{3 5 . 1}(+3.2)$ & 1.1 & $\mathbf{1 . 4}(+0.4)$ & 27.0 & $\mathbf{2 8 . 6}(+1.6)$ \\
\hline
\end{tabular}

Table 6. Soft Labeling with Latent Hierarchies for Linear Probing on ResNet-18. Instead of using WordNet to construct soft labels in Table 5, we adopted latent hierarchies constructed from pre-trained models using K-means clustering. Results show that using latent hierarchies also delivers a generalization boost compared to the baseline, although it is less significant than using WordNet. Experiments are listed here with the pro-OOD setting in Table 9.
generalization performance from derived soft labels. Table 10 reveals a moderate-strong correlation on ImageNet S/R/A, supported by visualizations in Fig. 8. The findings verify that a latent hierarchy derived from a more generalizable model (aligned closer to the WordNet hierarchy) provides higher quality in guiding the linear probe model training to be more generalizable. This visualization also shows that soft labels constructed from VLMs lead to better generalization. Since soft labels are derived from mean class feature clustering, this suggests that VLMs' superior generalization may stem from more regularized feature space distributions over encoded class centroids. Future work should explore the reasons behind VLMs' aligned feature spaces, potentially due to high-level language supervision.

\subsection*{4.3.3. Improving Generalization by Class Taxonomy Alignment with Prompt Engineering}

In this section, we discuss results on enhancing model generalization through prompt engineering in VLMs.

For VLM, integrating taxonomy-specific knowledge during zero-shot evaluation is straightforward. The WordNet hierarchy naturally indicates inter-class distances from class definitions. For example, 'dalmatian' and 'husky' are semantically close, both originating from the parent node 'dog'. We detail the results with CLIP-ViT32 (Radford et al., 2021) in Table 14. To test our hypothesis, we explicitly integrated hierarchical taxonomy relationships into the prompt for zero-shot VLM predictions. The prompt was designed as ' $\mathbf{A}$, which is a type of $\mathbf{B}$, which is a type of
$\mathbf{C}^{\prime}$, guiding the model to make taxonomy-aligned predictions. Additionally, we conducted two ablation studies: 1) Stack Parent: providing the correct taxonomy path without informing the model of the class name relationships; and 2) Shuffle Parent: informing the model of the hierarchical 'is-a' relationship but providing an incorrect taxonomy relationship randomly sampled from the tree. Our results demonstrate that informing the model of both the correct taxonomy and their hierarchical relationships significantly improves generalization. This improvement is evidenced by enhancements in Top-1 accuracy and test-time CrossEntropy (CE) across all datasets for all tested models.

\section*{5. Conclusions}

This work revitalizes the use of LCA distance, leveraging class taxonomies such as WordNet, to indicate model OOD performance. We assess the severity of model mispredictions in a manner agnostic to model modality, architecture or training data source, establishing a comprehensive metric for evaluating model generalization. Our findings across multiple ImageNet-OOD datasets highlight the superiority of LCA distance in reflecting the generalization capabilities of models trained with either class labels (VMs) or captions (VLMs), surpassing the traditional reliance on indistribution Top-1 accuracy (Miller et al., 2021). To extend the application of LCA distance measurement to any dataset, we introduce a method for creating latent hierarchies using K-means clustering, showcasing the resilience of LCA distance regardless of the applied taxonomy or hierarchy. Additionally, we demonstrate that aligning model predictions with class taxonomies, through soft labels or prompt engineering, can enhance model generalization. Our results on demonstrating VLMs' lower LCA distance and better soft label construction offer new insights into VLMs' superior model generalization from a feature distribution perspective.

Future research could focus on providing theoretical justification for the LCA-on-the-Line framework. For instance, exploring causal discovery (Brouillard et al., 2020) methods on the ImageNet dataset to construct a causal graph between classes and underlying variables may offer a more accurate reflection of causal relationships between classes.

\section*{Acknowledgements}

Authors thank Deva Ramanan for insightful discussions, and Hualiang Wang for valuable feedback on the manuscript. The work was partially supported by the CMU Argo Research Center. Shu Kong is partially supported by the University of Macau (SRG2023-00044-FST).

\section*{Limitation}

While we benchmarked and used LCA based on class hierarchy to measure model generalization, the findings from this work indicate that it is not an effective indicator for datasets visually similar to in-distribution data (like ImageNet2, more discussion in Appendix B). For these datasets, in-distribution Top1 remains a strong indicator, which potentially limits the utility of LCA. Also, it is expected that LCA will show a weaker discrimination between models on datasets with small number of classes (like CIFAR (Krizhevsky et al.)).

\section*{Impact Statement}

This research aims to enhance our understanding of model generalization mechanisms. However, it is crucial to recognize its potential misuse, such as in guiding adversarial attacks that reduce the generalization capabilities of existing models. Although not the intended purpose of our research, the dual potential of our findings in model generalization underscores the need for robust, secure model development and the implementation of ethical guidelines for deploying this knowledge.

\section*{References}

Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez$\mathrm{Paz}, \mathrm{D}$. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Baek, C., Jiang, Y., Raghunathan, A., and Kolter, J. Z. Agreement-on-the-line: Predicting the performance of neural networks under distribution shift. Advances in Neural Information Processing Systems, 35:19274-19289, 2022.

Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., and Katz, B. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019.

Barz, B. and Denzler, J. Hierarchy-based image embeddings for semantic image retrieval. In 2019 IEEE winter conference on applications of computer vision (WACV), pp. 638-647. IEEE, 2019.

Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., and Lord, N. A. Making better mistakes: Leveraging class hierarchies with deep networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12506-12515, 2020.

Brouillard, P., Lachapelle, S., Lacoste, A., Lacoste-Julien, S., and Drouin, A. Differentiable causal discovery from interventional data. Advances in Neural Information Processing Systems, 33:21865-21877, 2020.

Bühlmann, P. Invariance, causality and robustness. 2020.
Chen, M., Goel, K., Sohoni, N. S., Poms, F., Fatahalian, K., and Ré, C. Mandoline: Model evaluation under distribution shift. In International Conference on Machine Learning, pp. 1617-1629. PMLR, 2021.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. arXiv preprint arXiv:2212.07143, 2022.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818-2829, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. Ieee, 2009a.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and FeiFei, L. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR, 2009b.

Deng, W., Gould, S., and Zheng, L. On the strong correlation between model invariance and generalization. arXiv preprint arXiv:2207.07065, 2022.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth $16 \times 16$ words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Fang, A., Ilharco, G., Wortsman, M., Wan, Y., Shankar, V., Dave, A., and Schmidt, L. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning, pp. 6216-6234. PMLR, 2022.

Frome, A., Corrado, G. S., Shlens, J., Bengio, S., Dean, J., Ranzato, M., and Mikolov, T. Devise: A deep visualsemantic embedding model. Advances in neural information processing systems, 26, 2013.

Gare, G. R., Fox, T., Lowery, P., Zamora, K., Tran, H. V., Hutchins, L., Montgomery, D., Krishnan, A., Ramanan, D. K., Rodriguez, R. L., et al. Learning generic lung ultrasound biomarkers for decoupling feature extraction from downstream tasks. arXiv preprint arXiv:2206.08398, 2022.

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231, 2018.

HaoChen, J. Z., Wei, C., Gaidon, A., and Ma, T. Provable guarantees for self-supervised deep learning with spectral contrastive loss. Advances in Neural Information Processing Systems, 34:5000-5011, 2021.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In $C V P R, 2016$.

Hendrycks, D. and Dietterich, T. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.

Hendrycks, D. and Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In ICLR, 2017.

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340-8349, 2021.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1314-1324, 2019.

Hu, H., Sener, O., Sha, F., and Koltun, V. Drinking from a firehose: Continual learning with web-scale natural language. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In

Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708, 2017.

Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., and Keutzer, K. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and $<0.5 \mathrm{mb}$ model size. arXiv preprint arXiv:1602.07360, 2016.

Jabri, A., Joulin, A., and Van Der Maaten, L. Revisiting visual question answering baselines. In European conference on computer vision, pp. 727-739. Springer, 2016.

Jiang, Y., Nagarajan, V., Baek, C., and Kolter, J. Z. Assessing generalization of sgd via disagreement. arXiv preprint arXiv:2106.13799, 2021.

Kaur, J. N., Kiciman, E., and Sharma, A. Modeling the datagenerating process is necessary for out-of-distribution generalization. arXiv preprint arXiv:2206.07837, 2022.

Krizhevsky, A., Nair, V., and Hinton, G. Cifar-10 (canadian institute for advanced research). URL http://www. cs.toronto.edu/~kriz/cifar.html.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84-90, 2017.

Larochelle, H., Erhan, D., and Bengio, Y. Zero-data learning of new tasks. In $A A A I$, volume 1, pp. 3, 2008.

Li, C., Zhang, B., Shi, J., and Cheng, G. Multi-level domain adaptation for lane detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4380-4389, 2022a.

Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., and Hoi, S. C. H. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34: 9694-9705, 2021.

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888-12900. PMLR, 2022b.

Lin, Z., Shi, J., Pathak, D., and Ramanan, D. The clear benchmark: Continual learning on real-world imagery. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Lin, Z., Pathak, D., Wang, Y.-X., Ramanan, D., and Kong, S. Continual learning with evolving class ontologies. Advances in Neural Information Processing Systems, 35: 7671-7684, 2022.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012-10022, 2021.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11976-11986, 2022.

Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., and Tegmark, M. Kan: Kolmogorovarnold networks. arXiv preprint arXiv:2404.19756, 2024.

Lomonaco, V. and Maltoni, D. Core50: a new dataset and benchmark for continuous object recognition. In Conference on Robot Learning, pp. 17-26. PMLR, 2017.

Makar, M., Packer, B., Moldovan, D., Blalock, D., Halpern, Y., and D'Amour, A. Causally motivated shortcut removal using auxiliary labels. In International Conference on Artificial Intelligence and Statistics, pp. 739-766. PMLR, 2022.

Miller, G. A. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39-41, 1995.

Miller, G. A., Beckwith, R., Fellbaum, C., Gross, D., and Miller, K. J. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4): 235-244, 1990.
Miller, J., Krauth, K., Recht, B., and Schmidt, L. The effect of natural distribution shift on question answering models. In International Conference on Machine Learning, pp. 6905-6916. PMLR, 2020.

Miller, J. P., Taori, R., Raghunathan, A., Sagawa, S., Koh, P. W., Shankar, V., Liang, P., Carmon, Y., and Schmidt, L. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In International Conference on Machine Learning, pp. 77217735. PMLR, 2021.

Mustafa, B., Riquelme, C., Puigcerver, J., Pinto, A. S., Keysers, D., and Houlsby, N. Deep ensembles for lowdata transfer learning. arXiv preprint arXiv:2010.06866, 2020.

Peri, N., Dave, A., Ramanan, D., and Kong, S. Towards longtailed 3d detection. In Conference on Robot Learning, 2023.

Peters, J., Bühlmann, P., and Meinshausen, N. Causal inference by using invariant prediction: identification and confidence intervals. Journal of the Royal Statistical Society. Series B (Statistical Methodology), pp. 947-1012, 2016.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748-8763. PMLR, 2021.

Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., and Dollár, P. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428-10436, 2020.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pp. 5389-5400. PMLR, 2019.

Redmon, J. and Farhadi, A. Yolo9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7263-7271, 2017.

Roelofs, R., Shankar, V., Recht, B., Fridovich-Keil, S., Hardt, M., Miller, J., and Schmidt, L. A meta-analysis of overfitting in machine learning. Advances in Neural Information Processing Systems, 32, 2019.

Santurkar, S., Tsipras, D., and Madry, A. Breeds: Benchmarks for subpopulation shift. arXiv preprint arXiv:2008.04859, 2020.
Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings of the IEEE, 109(5): 612-634, 2021.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.

Shankar, V., Roelofs, R., Mania, H., Fang, A., Recht, B., and Schmidt, L. Evaluating machine accuracy on imagenet. In International Conference on Machine Learning, pp. 8634-8644. PMLR, 2020.

Shen, X., Liu, F., Dong, H., Lian, Q., Chen, Z., and Zhang, T. Weakly supervised disentangled generative causal representation learning. Journal of Machine Learning Research, 23:1-55, 2022.

Shi, Z., Carlini, N., Balashankar, A., Schmidt, L., Hsieh, C.-J., Beutel, A., and Qin, Y. Effective robustness against natural distribution shifts for models with different training data. arXiv preprint arXiv:2302.01381, 2023.

Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

Sturm, B. L. A simple method to determine if a music information retrieval system is a "horse". IEEE Transactions on Multimedia, 16(6):1636-1644, 2014.

Subramanian, J., Annadani, Y., Sheth, I., Ke, N. R., Deleu, T., Bauer, S., Nowrouzezahrai, D., and Kahou, S. E. Learning latent structural causal models. arXiv preprint arXiv:2210.13583, 2022.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826, 2016.

Tan, M. and Le, Q. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105-6114. PMLR, 2019.

Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., and Le, Q. V. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2820-2828, 2019.

Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B., and Schmidt, L. Measuring robustness to natural distribution shifts in image classification. Advances in Neural Information Processing Systems, 33:18583-18599, 2020.

Torralba, A. and Efros, A. A. Unbiased look at dataset bias. In CVPR 2011, pp. 1521-1528. IEEE, 2011.

Valmadre, J. Hierarchical classification at multiple operating points. arXiv preprint arXiv:2210.10929, 2022.

Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., and Belongie, S. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8769-8778, 2018.

Wan, A., Dunlap, L., Ho, D., Yin, J., Lee, S., Jin, H., Petryk, S., Bargal, S. A., and Gonzalez, J. E. Nbdt: neural-backed decision trees. arXiv preprint arXiv:2004.00221, 2020.

Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Lopes, R. G., Hajishirzi, H., Farhadi, A., Namkoong, H., et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 79597971, 2022.

Yang, M., Liu, F., Chen, Z., Shen, X., Hao, J., and Wang, J. Causalvae: Disentangled representation learning via neural structural causal models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9593-9602, 2021.

Zagoruyko, S. and Komodakis, N. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107115, 2021.

Zhang, X., Zhou, X., Lin, M., and Sun, J. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6848-6856, 2018.

\section*{A. Model Architectures}

We list all models used in ours experiment as follows, including 36 Vision Only Models (VMs) and 39 Vision-Language Models (VLMs).
\begin{tabular}{|c|c|c|c|}
\hline Model Category & Architecture & Number of models & Checkpoint Link \\
\hline \multirow{16}{*}{VM (Vision-Only-Models)} & AlexNet (Krizhevsky et al., 2017) & 1 & alexnet \\
\hline & ConvNeXt (Liu et al., 2022) & 1 & convnext_tiny \\
\hline & DenseNet (Huang et al., 2017) & 4 & \begin{tabular}{l}
densenet 121 \\
densenet 161 \\
densenet 169 \\
densenet201
\end{tabular} \\
\hline & EfficientNet (Tan \& Le, 2019) & 1 & efficientnet_b0 \\
\hline & GoogLeNett (Szegedy et al., 2015) & 1 & googlenet \\
\hline & Inceptionv3 (Szegedy et al., 2016) & 1 & inceptionV3 \\
\hline & MnasNet (Tan et al., 2019) & 4 & \begin{tabular}{l}
mnasnet0.5 \\
mnasnet0.75 \\
mnasnet 1.0 \\
mnasnet 1.3
\end{tabular} \\
\hline & Mobilenet-V3 (Howard et al., 2019) & 2 & mobilenetv3_small mobilenetv3_large \\
\hline & Regnet (Radosavovic et al., 2020) & 1 & regnet_y_1_6gf \\
\hline & Wide ResNet (Zagoruyko \& Komodakis, 2016) & 1 & wide_resnet101_2 \\
\hline & ResNet (He et al., 2016) & 5 & \begin{tabular}{l}
resnet18 \\
resnet 34 \\
resnet50 \\
resnet101 \\
resnet 152
\end{tabular} \\
\hline & ShuffleNet (Zhang et al., 2018) & 1 & shufflenet_v2_x2_0 \\
\hline & SqueezeNet (Iandola et al., 2016) & 2 & \[
\begin{aligned}
& \hline \text { squeezenet1_0 } \\
& \text { squeezenet1_1 }
\end{aligned}
\] \\
\hline & Swin Transformer (Liu et al., 2021) & 1 & swin_b \\
\hline & VGG (Simonyan \& Zisserman, 2015) & 8 & \begin{tabular}{l}
vgg11 \\
vgg 13 \\
vgg 16 \\
vgg 19 \\
vgg11_bn \\
vgg13_bn \\
vgg16_bn \\
vgg19_bn
\end{tabular} \\
\hline & ViT (Dosovitskiy et al., 2020) & 2 & \[
\begin{aligned}
& \text { vit_b_32 } \\
& \text { vit_1_32 }
\end{aligned}
\] \\
\hline \multirow{4}{*}{VLM (Vision-Language-Models)} & ALBEF (Li et al., 2021) & 1 & albef_feature_extractor \\
\hline & BLIP (Li et al., 2022b) & 1 & blip_feature_extractor_base \\
\hline & CLIP (Radford et al., 2021) & 7 & \begin{tabular}{l}
RN50 \\
RN101 \\
RN50x4 \\
ViT-B-32.pt \\
ViT-B-16.pt \\
ViT-L-14.pt \\
ViT-L-14-336px
\end{tabular} \\
\hline & OpenCLIP (Cherti et al., 2023) & 30 & ```
openCLIP:
openCLIP_('RN101', 'openai')
openCLIP_('RN101', 'yfcc15m')
openCLIP_('RN101-quickgelu', 'openai')
openCLIP_('RN101-quickgelu', 'yfcc 15m')
openCLIP_('RN50', 'cc12m')
openCLIP_('RN50', 'openai')
openCLIP_('RN50', 'yfcc15m')
openCLIP_('RN50-quickgelu', 'cc12m')
openCLIP_('RN50-quickgelu', 'openai')
openCLIP_('RN50-quickgelu', 'yfcc 15m')
openCLIP_('RN50x16', 'openai')
openCLIP_('RN50x4', 'openai')
openCLIP_('RN50x64', 'openai')
openCLIP_('ViT-B-16', 'laion2b_s34b_b88k')
openCLIP_('ViT-B-16', 'laion400m_e31')
openCLIP_('ViT-B-16', 'laion400m_e32')
openCLIP_('ViT-B-16-plus-240', 'laion400m_e31')
openCLIP_('ViT-B-16-plus-240', 'laion400m_e32')
openCLIP_('ViT-B-32', 'laion2b_e16')
openCLIP_('ViT-B-32', 'laion2b_s34b_b79k')
openCLIP_('ViT-B-32', 'laion400m_e31')
openCLIP_('ViT-B-32', 'laion400m_e32')
openCLIP_('ViT-B-32', 'openai')
openCLIP_('ViT-B-32-quickgelu', 'laion 400 m _e 31 ')
openCLIP_('ViT-B-32-quickgelu', 'laion 400 m _e 32 ')
openCLIP_('ViT-L-14', 'laion2b_s32b_b82k')
openCLIP_('ViT-L-14', 'laion400m_e31')
openCLIP_('ViT-L-14', 'laion 400 m _e 32 ')
openCLIP_('coca_ViT-B-32', 'laion2b_s13b_b90k')
openCLIP_('coca_ViT-L-14', 'laion2b_s13b_b90k')
``` \\
\hline
\end{tabular}

\section*{B. Discussion}

Reestablishing LCA as a Comprehensive Measure of Model Generalization. While Top 1 ID accuracy (Miller et al., 2021) demonstrates a clear linear trend with OOD datasets in models with similar training mechanisms, this relationship becomes less distinct across VMs and VLMs. This finding, echoed in earlier studies (Fang et al., 2022; Wortsman et al., 2022; Cherti et al., 2022), suggests a more nuanced understanding of how zero-shot VLMs with lower Top-1 accuracy can outperform competitive vision models in generalizing to unfamiliar datasets. While previous works have emphasized the significant impact of data diversity on generalization (Fang et al., 2022; Schuhmann et al., 2022; Kaur et al., 2022), our results indicate that the LCA offers a more all-encompassing assessment of model generalization. By considering factors such as training data size, architecture, loss, and others, LCA better measures a model's ability to accurately capture semantic distinctions common across ID and OOD benchmarks. This establishes a comprehensive benchmark that encompasses various generalization factors, addressing the issue of inflated VLM effectiveness on "Effective Robustness (Taori et al., 2020)". Future research should delve into large-scale analytic studies of generalization factors in conjunction with LCA.

ImageNet-v2 Demonstrates Similar Class Discrimination Features to ImageNet. ImageNet-v2, a recollection of ImageNet, is often used as an OOD dataset for ImageNet-based studies (Shankar et al., 2020; Miller et al., 2021; Baek et al., 2022). Our experiments indicate that ImageNet-v2 more closely resembles ImageNet than other OOD datasets. We hypothesize that the minimal external intervention in ImageNet-v2's data collection process results in visual similarities to ImageNet (as ImageNet-v2 is a recollection of ImageNet), allowing even spurious relationships encoded on ImageNet to transfer successfully to ImageNet-v2. Consequently, models pretrained on ImageNet (VMs) inflate accuracy on ImageNet-v2, disrupting the alignment with trends observed in VLMs.

Is it Possible for a Semantically-Aware (Low LCA) Model to Have Low Top 1 Accuracy? Our empirical analysis indicates a correlation: models not specifically tuned on class taxonomy, with lower Top 1 accuracy, tend to exhibit higher LCA distances. However, this relationship is correlational rather than causal. It remains feasible to design a model adversarially so it consistently predicts the semantically nearest class to the true class. In such cases, the model would show a low LCA distance while maintaining zero Top 1 accuracy. Therefore, while a correlation exists between Top 1 accuracy and LCA, causality cannot be inferred, and this relationship can be disrupted under deliberate adversarial training.

Does ImageNet LCA (Taxonomic Distance) Reflect ImageNet Top 1 Accuracy? It is often suggested that LCA and Top-1 accuracy exhibit similar trends on the same dataset (Deng et al., 2009b; Bertinetto et al., 2020). Intuitively, a high-performing model better fits the data distribution, leading to fewer severe errors. This pattern generally holds true for models under similar settings (either VM or VLM separately). However, when considering both VM and VLM models, ImageNet and ImageNet-v2 exhibit only a weak correlation between LCA and Top-1 accuracy, whereas other semantically distinct OOD datasets show a stronger relationship (validate in Section F.1). This finding challenges the prevailing belief that in-distribution Top-1 accuracy and LCA maintain the same ranking (Deng et al., 2009a; Bertinetto et al., 2020).

Why do we observe low LCA correlation numbers between IID test sets? From previous experiments, we observe that ImageNet LCA (Taxonomic Distance) does not correlate strongly with ImageNet/ImageNet-v2 Top-1 Accuracy, often showing a weak relationship, as illustrated in Figure 9. We hypothesize that this is due to ID accuracy inflation. In our LCA-on-the-Line framework, LCA is expected to be an unbiased measure of alignment to the class hierarchy. However, the VMs used in this work are trained on ImageNet and tend to 'inflate' ID accuracy when evaluated on IID test sets. As indicated in the bottom right two images of Figure 9, this inflation might causes datapoints to 'shift' in the direction of the red arrow, disrupting the unbiased linear relationship seen in VLMs that were not trained directly on ImageNet. Consequently, we should expect models evaluating LCA on unseen datasets to form a linear relationship, similar to the observed relationship on the other four severely shifted OOD datasets in Figure 9. Please refer to Section F. 1 and Table 13 for a numerical comparison.

\section*{C. LCA Illustration with Simulated Data}

To illustrate the hypotheses in Section 3: 1) Transferable features are more likely to be supported by the hierarchy and shared among neighboring classes; 2) Confounding features are less supported by the hierarchy and tend to appear in less relevant classes that are often more distant in the hierarchy; 3) LCA is useful in identifying features supported by the hierarchy, we created a simple example using a simulated dataset.

Consider a feature space $\mathbf{x}:=\left(x_{1}, x_{2}, x_{3}\right) \in \mathbb{R}^{3}$ and a latent class $z \in 1,2,3,4$, where class 1 and 2 are similar, and class 3 and 4 are similar. By design, we set the joint distribution of $\mathbf{x}$ and $z$ to follow a mixture of Gaussians, where $x_{1} \in\{1,3,15,17\}, x_{2} \in\{1,17,7,21\}, x_{3} \in\{0,0,0,0\}$ for each class respectively.
\[
\begin{array}{ll}
\mathbf{x} \mid z=1 \sim N\left(\mu_{1}, \mathbf{I}\right), & \mu_{1}=(1,1,0) \\
\mathbf{x} \mid z=2 \sim N\left(\mu_{2}, \mathbf{I}\right), & \mu_{2}=(3,17,0) \\
\mathbf{x} \mid z=3 \sim N\left(\mu_{3}, \mathbf{I}\right), & \mu_{3}=(15,7,0) \\
\mathbf{x} \mid z=4 \sim N\left(\mu_{4}, \mathbf{I}\right), & \mu_{4}=(17,21,0)
\end{array}
\]

Given a hierarchy preserving class proximity: root: (class 1, class 2 ), (class 3, class 4), by design, only feature $x_{1}$ supports the class hierarchy, as the distance w.r.t $x_{1}$ between classes $1 \& 2$ and classes $3 \& 4$ is smaller than those for other pairs. Feature $x_{2}$ can distinguish all four classes but is not supported by the class hierarchy. Feature $x_{3}$ is random noise with no predictive power for the latent class.

For the in-distribution (ID) data, all three features are observed, while for the out-of-distribution (OOD) data, only $x_{1}$ and $x_{3}$ are observed. From hypothesis in section $3, x_{1}$ can be considered a transferable causal feature because it is supported by the true class hierarchy and is observable in all datasets. In contrast, $x_{2}$ is a non-transferable confounding feature that does not preserve the class hierarchy and is only observable in the ID data. By design (larger $\mu$ gap between classes), confounder $x_{2}$ display stronger discrimination among four classes than $x_{1}$ on ID data.
We trained two logistic regression models on the in-distribution (ID) dataset, mimicking models that captured different features as predictive variables learned from the training data.
- Model $f$, which trains on the transferable causal feature $x_{1}$, and noise feature $x_{3}$.
- Model $g$, which trains on the non-transferable confounding feature $x_{2}$, and noise feature $x_{3}$.

From simulations ( 10,000 samples across 100 independent trials), we observed the following results listed in Table 7:
- Model $g$ achieved better ID accuracy because it can leverage $x_{2}$, which distinguishes all four classes effectively in the ID data.
- Model $f$ had better OOD accuracy because $x_{1}$ is a transferable feature that is also present in the OOD data, supported by the true class hierarchy that's invariant across ID and OOD data.
- Model $f$ showed better (lower) LCA distance on the ID test set, indicating that it captures the class hierarchy better by relying on the transferable causal feature $x_{1}$.

This example illustrates the hypothesis presented in Section 3 and provides the expected output in Table 7. The results suggest that LCA can effectively identify models that capture relationships aligned with the hierarchical structure. For further details, please refer to code snippet.

\section*{D. Metric}

In this section, we outline the metrics adopted for our experiment.

\section*{D.1. Correlation Measurement}

Correlation measurements quantify the degree of association between two variables. This can be further subdivided into linearity and ranking measurements.

\section*{D.1.1. Linearity Measurement}

Linearity measurement evaluates the strength and direction of a linear relationship between two continuous variables. We use the $\mathrm{R}^{\mathbf{2}}$ and Pearson correlation coefficients to assess linearity.
\begin{tabular}{lccc}
\hline & ID Top1 Error $\downarrow$ & ID LCA Distance $\downarrow$ & OOD Top1 Error $\downarrow$ \\
\hline g(w. confounding feature) & $\mathbf{0 . 1 4 2 3}$ & 2.000 & 0.7503 \\
f(w. transferable feature) & 0.3287 & $\mathbf{1 . 0 0 5}$ & $\mathbf{0 . 3 1 9 7}$ \\
Diff & +0.1864 & -0.995 & -0.4306 \\
\hline
\end{tabular}

Table 7. Observation from simulation data with 100 trials. The average ID test accuracy error (i.e. top 1 error) ID_Top1_Error $\downarrow$, ID test LCA distance ID_LCA_Distance $\downarrow$, and OOD test accuracy error OOD_Top1_Error $\downarrow$ for generalizable "good" prediction model $f$ and non-generalizable "bad" prediction model $g$ over 100 independent trials. Specifically, we design the data generation process as described in (1), and $f$ is "good" as it learns to rely on the transferable causal features supported by hiearachy; while $g$ is "bad" as it instead relies on the non-transferable confounding features not supported by hiearachy. In this example, ID LCA distance is a better indicator of OOD performance than ID Top1 accuracy, and model f display better generalization to OOD dataset despite lower ID Top 1 accuracy.
$\mathbf{R}^{\mathbf{2}}$ (Coefficient of determination): The $\mathrm{R}^{2}$, or coefficient of determination, quantifies the proportion of the variance in the dependent variable that can be predicted from the independent variable(s). It ranges from 0 to 1 , where 1 indicates perfect predictability. It is defined as:
\[
R^{2}=1-\frac{\sum_{i=1}^{n}\left(y_{i}-f\left(x_{i}\right)\right)^{2}}{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}}
\]
where $f\left(x_{i}\right)$ is the prediction of $y_{i}$ from the model, $\bar{y}$ is the mean of the actual $y$ values, and $n$ is the number of data points.
PEA (Pearson correlation coefficient): The Pearson correlation coefficient, denoted as $r$, measures the linear relationship between two datasets. It is defined as:
\[
r=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}}}
\]
where $\bar{x}$ and $\bar{y}$ are the mean values of the datasets $x$ and $y$, respectively, and $n$ is the number of data points.

\section*{D.1.2. RANKING MEASUREMENT}

Ranking measurement evaluates the degree of correspondence between the rankings of two variables, even when their relationship is non-linear. The Kendall and Spearman rank correlation coefficients are metrics used for this purpose.

KEN (Kendall rank correlation coefficient): Also known as Kendall's tau ( $\tau$ ), this coefficient measures the ordinal association between two variables. It is defined as:
\[
\tau=\frac{\text { (number of concordant pairs) }-(\text { number of discordant pairs })}{\frac{1}{2} n(n-1)}
\]
where $n$ is the number of data points.
SPE (Spearman rank-order correlation coefficient): The Spearman rank-order correlation coefficient, denoted as $\rho$, assesses the monotonic relationship between two variables. It is defined as:
\[
\rho=1-\frac{6 \sum_{i=1}^{n} d_{i}^{2}}{n\left(n^{2}-1\right)}
\]
where $d_{i}$ is the difference between the ranks of corresponding data points in the two datasets and $n$ is the number of data points.

\section*{D.2. Taxonomy Measurement}

Taxonomy measurement is designed to assess the alignment between the model-predicted class ranking and the predefined class taxonomy hierarchy tree. This is also referred to as 'mistake severity' or 'taxonomic distance'.

\section*{D.2.1. LCA DISTANCE}

Following (Bertinetto et al., 2020; Valmadre, 2022), we define LCA distance using a predefined hierarchy tree, as indicated in Fig. 3. We adopt class distance in a hierarchical tree format to denote inter-class relationships, which is necessary to
calculate LCA and ELCA (cf. Section D.3). Given a ground-truth node y (node 1 in the plot), a model prediction node $y^{\prime}$, and their lowest common ancestor class node $N_{L C A}\left(y, y^{\prime}\right)$. We define it as:
\[
D_{L C A}\left(y^{\prime}, y\right):=f(y)-f\left(N_{L C A}\left(y, y^{\prime}\right)\right)
\]
where $f(\cdot)$ represents a function for a node's score, such as the tree depth or information content.
Scores as tree depths: We define a function $P(x)$ to retrieve the depth of node x from tree T. Then, LCA distance is defined as:
\[
D_{L C A}^{P}\left(y^{\prime}, y\right):=\left(P(y)-P\left(N_{L C A}\left(y^{\prime}, y\right)\right)\right)+\left(P\left(y^{\prime}\right)-P\left(N_{L C A}\left(y^{\prime}, y\right)\right)\right)
\]
where we also append $\left(P\left(y^{\prime}\right)-P\left(N_{L C A}\left(y^{\prime}, y\right)\right)\right)$ to counter tree imbalance.
Scores as information: Defining score as tree depth may be vulnerable to an imbalanced hierarchical tree. Thus, we also define a node's score as information to put more weight on nodes with more descendants. Formally, following (Valmadre, 2022), we apply a uniform distribution $p$ to all leaf nodes in the tree that indicate a class in the classification task. The probability of each intermediate node in the tree is calculated by recursively summing the scores of its descendants. Then, the information of each node is calculated as $I($ node $):=-\log 2(p)$. The LCA distance is then defined as:
\[
D_{L C A}^{I}\left(y^{\prime}, y\right):=I(y)-I\left(N_{L C A}\left(y^{\prime}, y\right)\right)
\]

In this work, we adopt $D_{L C A}^{I}\left(y^{\prime}, y\right)$ for LCA measurements, and $D_{L C A}^{P}\left(y^{\prime}, y\right)$ for linear probing experiments.

\section*{D.3. ELCA distance}

For a sample $X_{i}$ whose ground-truth class is $y_{i}$, and the model outputs ( $\widehat{p}_{1, i}, \ldots, \widehat{p}_{K, i}$ ) over the $K$ classes (e.g., 1000 in ImageNet), we define the Expected Lowest Common Ancestor Distance (ELCA):
\[
D_{E L C A}(\operatorname{model}, \mathcal{M}):=\frac{1}{n K} \sum_{i=1}^{n} \sum_{k=1}^{K} \widehat{p}_{k, i} \cdot D_{L C A}\left(k, y_{i}\right)
\]

From a probabilistic perspective, $D_{E L C A}$ is a weighted measure of mistake severity according to the model's confidence in each node in the hierarchy. Intuitively, it combines the LCA distance with a cross-entropy measurement.

The proposed ELCA distance provides a more generalized metric for assessing model performance compared to Top 1 accuracy, LCA distance, and cross entropy. Top 1 accuracy only considers the top-ranked class; LCA distance measures the Top $n$ class rankings but treats each class equally (Bertinetto et al., 2020); Cross-entropy solely focuses on the model's assigned probability to the ground-truth class, and ELCA extends it to all classes. The ELCA distance captures the probabilistic distribution of mistake severity across all candidate classes.
For implementation, ELCA is a weighted combination of the LCA distance for each leaf node [1,2,3,4] as in Fig. 3, weighted by class probability. Formally, for each prediction node $X_{i}$, the probabilistic distribution over all candidate classes can be obtained by applying a softmax function $\operatorname{softmax}(x): \mathbb{R} \rightarrow[0,1]$ to get model outputs probability $\left(\widehat{p}_{1, i}, \ldots, \widehat{p}_{K, i}\right)$ over the $K$ classes (e.g., 1000 in ImageNet).

In Table 8, we also demonstrate that models with better OOD generalization (OOD Top 1 accuracy) usually also have lower LCA/ELCA distances.
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline \multirow[t]{2}{*}{Model} & \multicolumn{3}{|c|}{ImageNet} & \multicolumn{3}{|c|}{ImageNetv2} & \multicolumn{3}{|c|}{ImageNet-S} & \multicolumn{3}{|c|}{ImageNet-R} & \multicolumn{3}{|c|}{ImageNet-A} & \multicolumn{3}{|c|}{ObjectNet} \\
\hline & LCA & ELCA & Top1 & LCA & ELCA & Topl & LCA & ELCA & Topl & LCA & ELCA & Top1 & LCA & ELCA & Topl & LCA & ELCA & Topl \\
\hline ResNet18 (He et al., 2016) & 6.643 & 7.505 & 0.698 & 6.918 & 7.912 & 0.573 & 8.005 & 9.283 & 0.202 & 8.775 & 8.853 & 0.330 & 8.449 & 9.622 & 0.011 & 8.062 & 8.636 & 0.272 \\
\hline ResNet50 (He et al., 2016) & 6.539 & 7.012 & 0.733 & 6.863 & 7.532 & 0.610 & 7.902 & 9.147 & 0.235 & 8.779 & 8.668 & 0.361 & 8.424 & 9.589 & 0.018 & 8.029 & 8.402 & 0.316 \\
\hline CLIP_RN50 (Radford et al., 2021) & 6.327 & 9.375 & 0.579 & 6.538 & 9.442 & 0.511 & 6.775 & 9.541 & 0.332 & 7.764 & 9.127 & 0.562 & 7.861 & 9.526 & 0.218 & 7.822 & 8.655 & 0.398 \\
\hline CLIP_RN50x4 (Radford et al., 2021) & 6.166 & 9.473 & 0.641 & 6.383 & 9.525 & 0.573 & 6.407 & 9.518 & 0.415 & 7.435 & 8.982 & 0.681 & 7.496 & 9.388 & 0.384 & 7.729 & 8.354 & 0.504 \\
\hline
\end{tabular}

Table 8. Model performance corresponds to mistake severity. LCA $\downarrow /$ ELCA $\downarrow /$ Top $1 \uparrow$ indicate measurements on a given dataset. We present two pairs of model comparisons from the VMs and VLMs families with different generalization abilities. Note that ELCA should not be compared across modalities, as it is sensitive to logit temperature.

\section*{E. Experiment Setup}

\section*{E.1. K-mean Clustering for Latent Class Hierarchy Construction}

As depicted in Fig 6, we begin with a pretrained model $M$, in-distribution image data $X$, and labels $Y$ for $k$ classes. Initially, we extract the in-distribution data features $M(X)$. With known labels, we categorize $M(X)$ by $Y$, resulting in $k$ average class features, denoted as $k X$. Utilizing these per-class average features, we perform a 9 -layer hierarchical clustering. For $k X$, we apply the K-means algorithm, setting the number of cluster centers as $2^{i}$, where $i$ ranges from $1,2,3,4, \ldots, 9$ since $2^{9}<1000$ (ImageNet have 1000 classes). This procedure results in 9 cluster outcomes. Subsequently, we find the LCA node between each pair of the $k$ classes, to determine the cluster level at which both classes exists in the same cluster. We use the height of the common cluster as their pairwise LCA height to be retrieved at training/evaluation. By definition, all classes share a base cluster level of 10 .

\section*{E.2. Soft Loss for Hierarchy Alignment}

This section illustrates the loss function used in our linear probing experiment. For a dataset with $n$ classes, we first establish an $n \times n$ LCA distance matrix $M$ (visualize in Figure 7), where $M[i, k]$ indicates the pairwise LCA distance $D_{\mathrm{LCA}}(i, k)$, calculated using either the WordNet hierarchy or latent hierarchy derived from the K-means clustering (as introduced in the main paper). Next, we scale $M$ by applying a temperature term $T$, and finally apply MinMax scaling to normalize the values between 0 and 1 .
\[
M_{\mathrm{LCA}}=\operatorname{MinMax}\left(M^{T}\right)
\]

As shown in the code snippet below, we construct the auxiliary loss by assigning class likelihoods beyond the top-1 (one-hot), extending to all classes. Similar to adopting one-hot encoding to let the model focus on the top- 1 ground-truth, we use the reverse of LCA matrix as an alignment indicator, where ground-truth index have the largest value of 1 . This alignment can be applied to both BCE and CE types of loss. Details in our code.
```
Algorithm 1 LCA Alignment Loss
    function LCA_ALIGNMENT_LOSS(logits, targets, alignment_mode, LCA_matrix, lambda_weight=0.03)
        reverse_LCA_matrix $\leftarrow 1-$ LCA_matrix
        Compute predicted probabilities: probs $\leftarrow$ softmax (logits, dim=1)
        One-hot encode the targets: one_hot_targets
        Compute standard cross-entropy loss:
            standard_loss $\leftarrow-\sum($ one_hot_targets $\cdot \log ($ probs $)$, dim $=1)$
        if alignment_mode $==$ ' $B C E$ ' then
            criterion $\leftarrow$ BCEWithLogitsLoss (reduction= 'none')
            Compute soft loss:
            soft_loss $\leftarrow$ mean (criterion(logits, reverse_LCA_matrix[targets]), dim=1)
        else if alignment_mode $==$ ' $\mathrm{CE}^{\prime}$ then
            Compute soft loss:
            soft_loss $\leftarrow-$ mean (reverse_LCA_matrix[targets] $\cdot \log ($ probs $)$, dim $=1$ )
        end if
        total_loss $\leftarrow$ lambda_weight $\cdot$ standard_loss + soft_loss
        Return mean loss over the batch: return mean (total_loss)
    end function
```

For the experiments in the main paper, we set lambda $=0.03$, temperature $=25$, and use $C E$ as the soft loss. Note that a smaller lambda scales down the standard cross-entropy loss. We found that using a large temperature, which assign semantic-closer classes with a larger likelihood, boost model generalization better.

\section*{E.3. Ablation study: Using class ontology as soft labels}

In Table 9, we present ablation study on soft loss labels for linear probing from section 4.3.2.
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline & & ImgNet & ImgNet-V2 & ImgNet-S & ImgNet-R & ImgNet-A & ObjectNet \\
\hline & CE-only & 69.4 & 56.4 & 19.7 & 31.9 & 1.1 & 27.0 \\
\hline & CE + interpolation & 69.4 & 56.6 & 19.9 & 32.7 & 1.3 & 27.4 \\
\hline & (Ours) CE + Soft Loss (no ID accuracy drop) & 69.5 & 56.5 & 19.7 & 32.4 & 1.1 & 27.3 \\
\hline et 18 (He et al., 2016) & (Ours) CE + Soft Loss (pro-OOD) & 69.2 & 56.4 & 20.3 & 34.1 & 1.4 & 27.6 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 69.4 & 56.9 & 20.7 & 33.8 & 1.2 & 28.0 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 68.0 & 55.9 & 21.2 & 35.1 & 1.4 & 28.6 \\
\hline & CE-only & 79.5 & 67.9 & 25.5 & 36.5 & 10.3 & 43.2 \\
\hline & CE + interpolation & 79.5 & 67.8 & 25.6 & 36.6 & 10.6 & 43.3 \\
\hline & (Ours) CE + Soft Loss (no ID accuracy drop) & 79.8 & 68.6 & 27.7 & 42.5 & 16.2 & 45.5 \\
\hline ResNet 50 (He et al., 2016) & (Ours) CE + Soft Loss (pro-OOD) & 79.8 & 68.6 & 27.7 & 42.5 & 16.2 & 45.5 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 79.8 & 68.6 & 27.7 & 42.5 & 16.2 & 45.5 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 79.8 & 68.6 & 27.7 & 42.5 & 16.2 & 45.5 \\
\hline & CE-only & 75.8 & 62.9 & 27.0 & 40.5 & 8.0 & 27.6 \\
\hline & CE + interpolation & 75.7 & 62.4 & 27.0 & 40.5 & 8.2 & 27.7 \\
\hline & (Ours) CE + Soft Loss (no ID accuracy drop) & 75.8 & 62.7 & 26.9 & 40.4 & 8.2 & 27.8 \\
\hline VIT-B (Dosovitskiy et al., 2020) & (Ours) CE + Soft Loss (pro-OOD) & 75.4 & 62.4 & 28.0 & 42.2 & 9.1 & 27.9 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 75.9 & 62.8 & 27.6 & 41.5 & 8.6 & 28.1 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 75.4 & 62.4 & 28.0 & 42.2 & 9.1 & 27.9 \\
\hline & CE-only & 76.8 & 63.9 & 28.4 & 42.2 & 10.6 & 28.7 \\
\hline & & 76.7 & 64.0 & 28.3 & 42.1 & 10.9 & 28.9 \\
\hline & (Ours) CE + Soft Loss (no ID accuracy drop) & 76.8 & 64.1 & 28.4 & 42.2 & 10.5 & 28.7 \\
\hline VIT & (Ours) CE + Soft Loss (pro-OOD) & 76.7 & 63.6 & 29.4 & 43.9 & 11.7 & 29.0 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 76.8 & 63.8 & 29.2 & 43.6 & 11.5 & 29.0 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 76.7 & 63.6 & 29.4 & 43.9 & 11.7 & 29.0 \\
\hline & CE-only & 82.0 & 70.6 & 28.7 & 42.4 & 21.8 & 44.4 \\
\hline & CE + interpolation & 82.0 & 70.8 & 28.8 & 42.3 & 22.2 & 44.7 \\
\hline Con & (Ours) CE + Soft Loss (no ID accuracy drop) & 82.0 & 70.7 & 28.7 & 42.3 & 21.9 & 44.6 \\
\hline Con & (Ours) CE + Soft Loss (pro-OOD) & 81.8 & 71.1 & 30.4 & 44.8 & 26.3 & 45.7 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 82.1 & 71.0 & 30.0 & 44.3 & 25.2 & 45.5 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 81.8 & 71.1 & 30.4 & 44.8 & 26.3 & 45.7 \\
\hline & CE-only & 83.1 & 72.0 & 30.3 & 43.5 & 29.5 & 48.3 \\
\hline & CE + interpolation & 83.1 & 71.8 & 30.4 & 43.7 & 29.9 & 48.3 \\
\hline & (Ours) CE + Soft Loss (no ID accuracy drop) & 83.2 & 72.0 & 31.0 & 44.2 & 30.9 & 49.0 \\
\hline Swin Transformer (Liu et al., 2021) & (Ours) CE + Soft Loss (pro-OOD) & 83.0 & 71.8 & 31.6 & 45.5 & 33.3 & 49.4 \\
\hline & (Ours) CE + Soft Loss + interpolation (no ID accuracy drop) & 83.2 & 71.9 & 31.4 & 45.3 & 32.7 & 49.5 \\
\hline & (Ours) CE + Soft Loss + interpolation (pro-OOD) & 83.0 & 71.8 & 31.6 & 45.5 & 33.3 & 49.4 \\
\hline
\end{tabular}

Table 9. Ablation Study on Soft Loss Labels for Linear Probing from Section 4.3.2. CE-only: model trained with Cross-Entropy (CE) loss only, as a baseline; Soft Loss: soft label loss generated from hierarchy; Interpolation: linear interpolation in weight space between CE-only and the current method; No ID Accuracy Drop: models that do not introduce an accuracy drop on ImageNet (ID) compared to the baseline (CE-only); Pro-OOD: models with parameters that prefer the improvement of OOD generalization, even at the cost of a slight ID accuracy drop, to demonstrate the potential of our methods in enhancing generalization. Note that some models might be selected in multiple settings and appear in multiple rows. Results show that 1). Incorporating soft labels significantly enhances OOD performance across all network architectures without sacrificing ID accuracy. 2). Weight interpolation further boosts OOD generalization, particularly in models supervised with soft labels. 3). Tuning the weight interpolation allows for a balance between maintaining ID accuracy and further improving OOD performance, demonstrating the method's flexibility and practicality.

\section*{E.4. Does the Generalization Quality of the Pretrained Source Model Affect the Quality of Soft Labels?}

This section continues the discussion in Section 4.3.2. We present our findings in Table 10 and Figure 8. The results reveal a moderate-strong correlation between the ID LCA of the pretrained source model, and the generalization capabilities of the linear probe model trained from the source-model-derived latent hierarchy.
![](assets/asset_7.jpg)

Figure 7. Visualization of pair-wise LCA distance for ImageNet classes. Each row signifies the LCA distance between a specific class and the reference class, arranged in ascending order, with the diagonal index indicating the shortest distance. From left to right: WordNet hierarchy; matrix constructed from ResNet50 (He et al., 2016); and matrix constructed from CLIP ResNet50 (Radford et al., 2021).
\begin{tabular}{lllllll}
\hline & ImageNet & ImageNetv2 & ImageNet-S & ImageNet-R & ImageNet-A & ObjectNet \\
\hline \multirow{2}{*}{ Corr(ID LCA, Soft Labels Quality } & $\frac{\text { PEA }}{0.187}$ & $\underline{\text { PEA }}$ & $\underline{\text { PEA }}$ & PEA & $\underline{\text { PEA }}$ & $\frac{\text { PEA }}{0.301}$
\end{tabular}

Table 10. Correlation Measurement between Source Model Generalization Ability and Soft Labels Quality. Following the K-Means clustering algorithm, we constructed 75 LCA distance matrices (class hierarchies) from 75 pretrained source models on ImageNet. We then used these LCA distance matrices as soft labels to guide linear probing over ResNet-18 features (as described in Section 4.3.2). The table indicates a moderate-strong correlation between the in-distribution LCA of the pretrained source model and the out-of-distribution (OOD) accuracy on the linear probe model using the corresponding derived LCA distance matrix. Visualization is shown in Figure 8.

\section*{E.5. Hyperparameters and Computational Resources}

In the linear probing experiment, we chose hyperparameters based on the task at hand. The learning rate was set to 0.001 , with a batch size of 1024 . We used the AdamW optimizer with weight decay and a cosine learning rate scheduler with a warm-up iteration. The warm-up type was set to 'linear' with a warm-up learning rate of $1 \mathrm{e}-5$. The experiment was run for 50 epochs. For our computational resources, we utilized a single NVIDIA GeForce GTX 1080 Ti GPU.

\section*{F. Supplementary Results}

\section*{F.1. Does ImageNet LCA (Taxonomic Distance) Reflect ImageNet Top-1 Accuracy?}

Here, we present numerical results to support the discussion in Section B. We challenge the common belief that LCA and Top-1 accuracy follow parallel trends within the same dataset. As illustrated in Figures 9 and Table 13, when including both VMs and VLMs, ImageNet and ImageNet-v2 show a weak correlation between LCA and Top-1 accuracy within the same dataset. In contrast, other semantically distinct OOD datasets exhibit a stronger relationship. We provide a hypothesis in discussion section B on 'VMs ID accuracy inflation' to explain this.

\section*{F.2. Comprehensive Results from Main Paper}

Extended from Table 2 and Table 3 in the main paper, we present measurements on only-VMs and only-VLMs in Table 11 and Table 12, respectively. Similarly, LCA is also a very good OOD indicator when involving only VMs or VLMs.

\section*{F.3. Ranking Measurement of LCA-on-the-Line}

Here we present the numerical results for ranking measures in comparison to the commonly used Top-1 in-distribution accuracy in Table 15. Similarly, in-distribution LCA distance presents strong results in both preserving linearity and ranking.
![](assets/asset_8.jpg)

Figure 8. Correlation Measurement between Source Model Generalization Ability and Soft Labels Quality. y-axis: LCA distance on ImageNet (ID dataset) between WordNet hierarchy and each of the pretrained models (that generate hierarchies). 4 x -axis: top- 1 accuracy on an OOD dataset by linear probing over each of the generated hierarchies. This plot visualizes the results from Table 10. It shows a moderate-strong correlation between the two variables on ImageNet-S/R/A and ObjectNet(besides some noisy data points). It also indicates that latent hierarchies constructed from VLMs tend to cluster on the right side of the x -axis, suggesting better generalization compared to those from VMs.
![](assets/asset_9.jpg)

Table 11. Correlation measurement of ID LCA/Top1 with OOD Top1/Top5 on 75 models across modality following Fig 5. The 'ALL grouping' demonstrates that LCA has a strong correlation with OOD performance on all datasets (except ImageNet-v2). We take the absolute value of all correlations for simplicity. Equivalently, LCA is also a very good OOD indicator when only involved VM or VLM.
![](assets/asset_10.jpg)

Figure 9. Predicting LCA (VM+VLM, $\mathbf{7 5}$ models) on the same dataset As per Table 13. Each plot's x -axis represents dataset Top-1 accuracy, while the y-axis shows LCA distance measured on the same datasets. The plots reveal that ImageNet and ImageNet-v2 do not exhibit a strong correlation between LCA and Top-1 accuracy, in contrast to other semantically distinct OOD datasets. This observation challenges the common belief that in-distribution Top-1 accuracy and LCA distance maintain the same order (Deng et al., 2009a; Bertinetto et al., 2020). More details in discussion section B.
\begin{tabular}{lllllll}
\hline & & ImageNetv2 & ImageNet-S & ImageNet-R & ImageNet-A & ObjectNet \\
\hline ALL & ID Top1 (Miller et al., 2021) & $\mathbf{0 . 0 4 0}$ & 0.230 & 0.277 & 0.192 & 0.178 \\
& AC (Hendrycks \& Gimpel, 2017) & $\underline{0.043}$ & $\underline{0.124}$ & $\mathbf{0 . 1 1 3}$ & 0.324 & $\underline{0.127}$ \\
& Aline-D (Baek et al., 2022) & 0.121 & 0.270 & 0.167 & 0.409 & 0.265 \\
& Aline-S (Baek et al., 2022) & 0.072 & 0.143 & 0.201 & $\underline{0.165}$ & 0.131 \\
& (Ours) ID LCA & 0.162 & $\mathbf{0 . 0 9 3}$ & $\underline{0.114}$ & $\underline{\mathbf{0 . 1 0 3}}$ & $\mathbf{0 . 0 4 8}$ \\
\hline VLM & ID (Miller et al., 2021) & $\mathbf{0 . 0 1 4}$ & 0.077 & $\underline{0.064}$ & 0.127 & $\underline{0.052}$ \\
& AC (Hendrycks \& Gimpel, 2017) & $\underline{0.029}$ & $\mathbf{0 . 0 5 0}$ & $\mathbf{0 . 0 4 4}$ & 0.217 & 0.088 \\
& Aline-D (Baek et al., 2022) & 0.151 & 0.250 & 0.081 & 0.296 & 0.260 \\
& Aline-S (Baek et al., 2022) & 0.070 & $\underline{0.069}$ & 0.068 & $\mathbf{0 . 0 8 0}$ & 0.153 \\
& (Ours) ID LCA & 0.047 & 0.083 & 0.070 & $\underline{0.105}$ & $\mathbf{0 . 0 4 3}$ \\
\hline VM & ID (Miller et al., 2021) & $\mathbf{0 . 0 1 3}$ & $\mathbf{0 . 0 9 9}$ & $\underline{0.108}$ & $\mathbf{0 . 1 4 3}$ & $\underline{0.068}$ \\
& AC (Hendrycks \& Gimpel, 2017) & 0.059 & 0.204 & 0.188 & 0.441 & 0.168 \\
& Aline-D (Baek et al., 2022) & 0.083 & 0.427 & 0.313 & 0.665 & 0.364 \\
& Aline-S (Baek et al., 2022) & 0.105 & 0.182 & $\mathbf{0 . 0 9 2}$ & 0.574 & 0.216 \\
& (Ours) ID LCA & $\underline{0.029}$ & $\underline{0.102}$ & 0.113 & $\underline{0.145}$ & $\mathbf{0 . 0 6 5}$ \\
\hline
\end{tabular}

Table 12. Error Prediction of OOD Datasets across 75 models of diverse settings with MAE loss $\downarrow$. Top1 in bold and Top2 in underline. Despite ImageNet's in-distribution accuracy maintain as a significant indicator of ImageNet-v2 accuracy, the in-distribution LCA outperforms it as a robust error predictor across four naturally distributed OOD datasets.
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline Model & Group & \multicolumn{2}{|l|}{ImageNet} & \multicolumn{2}{|l|}{ImageNetv2} & \multicolumn{2}{|l|}{ImageNet-S} & \multicolumn{2}{|l|}{ImageNet-R} & \multicolumn{2}{|l|}{ImageNet-A} & \multicolumn{2}{|l|}{ObjectNet} \\
\hline \multirow{12}{*}{Top1->LCA} & \multirow{4}{*}{ALL} & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA \\
\hline & & $\overline{0.174}$ & $\overline{0.417}$ & $\overline{0.1} 14$ & $\overline{0.337}$ & $\overline{\mathbf{0 . 8 3 5}}$ & 0.914 & $\overline{0.770}$ & $\overline{0.878}$ & $\overline{0.851}$ & $\overline{0.923}$ & $\overline{0.657}$ & $\overline{\mathbf{0 . 8 1 0}}$ \\
\hline & & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE \\
\hline & & $\overline{0.280}$ & $\overline{0.266}$ & 0.237 & $\overline{0.294}$ & $\overline{\mathbf{0 . 8 1 8}}$ & $\overline{0.926}$ & 0.621 & $\overline{0.803}$ & $\overline{\mathbf{0 . 8 2 5}}$ & $\overline{0.951}$ & $\overline{0.673}$ & $\overline{0.823}$ \\
\hline & \multirow{4}{*}{VLM} & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA \\
\hline & & $\overline{0.938}$ & $\overline{0.969}$ & $\overline{0.891}$ & $\overline{0.944}$ & $\overline{0.945}$ & 0.972 & $\overline{0.878}$ & 0.937 & $\overline{0.725}$ & 0.851 & $\overline{0.510}$ & $\overline{0.714}$ \\
\hline & & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE \\
\hline & & 0.880 & 0.969 & 0.799 & 0.881 & 0.864 & 0.963 & 0.753 & 0.902 & 0.689 & 0.869 & 0.529 & 0.720 \\
\hline & \multirow{4}{*}{VM} & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA & $R^{2}$ & PEA \\
\hline & & $\overline{0.973}$ & 0.986 & $\overline{0.890}$ & $\overline{0.943}$ & $\overline{0.934}$ & 0.966 & $\overline{0.095}$ & $\overline{0.310}$ & $\overline{0.840}$ & $\overline{0.916}$ & $\overline{0.948}$ & $\overline{0.974}$ \\
\hline & & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE \\
\hline & & $\overline{0.911}$ & $\overline{0.980}$ & $\overline{0.758}$ & $\overline{0.910}$ & $\overline{0.854}$ & $\overline{0.963}$ & $\overline{0.149}$ & $\overline{0.222}$ & $\overline{\mathbf{0 . 8 3 9}}$ & $\overline{0.952}$ & $\overline{0.854}$ & $\overline{0.960}$ \\
\hline
\end{tabular}

Table 13. Correlation Measurement between Top-1 Accuracy and LCA on the Same Dataset. This analysis uses 75 models across different modalities ( 36 VMs and 39 VLMs ) on all six ImageNet datasets. While the main paper employs ID LCA to predict OOD performance (e.g., Corr(ImageNet LCA, ImageNet-A Top-1 Accuracy)), this setting differs by using LCA to predict Top-1 accuracy on the same dataset (e.g., Corr(ImageNet-A LCA, ImageNet-A Top-1 Accuracy)). Following Figure 9, we highlight strong correlation indications. For simplicity, we take the absolute value of all correlations. More details in discussion section B.
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline \multirow[t]{2}{*}{Model} & \multicolumn{2}{|r|}{ImgN} & \multicolumn{2}{|r|}{ImgN-v2} & \multicolumn{2}{|r|}{ImgN-S} & \multicolumn{2}{|r|}{ImgN-R} & \multicolumn{2}{|r|}{ImgN-A} & \multicolumn{2}{|r|}{ObjNet} \\
\hline & Top1 $\uparrow$ & Test CE $\downarrow$ & Top1 $\uparrow$ & Test CE $\downarrow$ & Top1 $\uparrow$ & Test CE $\downarrow$ & Top1 $\uparrow$ & Test CE $\downarrow$ & Top1 $\uparrow$ & Test CE $\downarrow$ & Top1 $\uparrow$ & Test CE $\downarrow$ \\
\hline Baseline & 0.589 & 9.322 & 0.517 & 9.384 & 0.379 & 9.378 & 0.667 & 8.790 & 0.294 & 9.358 & 0.394 & 8.576 \\
\hline Stack Parent & 0.381 & 9.389 & 0.347 & 9.395 & 0.219 & 9.561 & 0.438 & 9.258 & 0.223 & 9.364 & 0.148 & 9.076 \\
\hline Shuffle Parent & 0.483 & 9.679 & 0.432 & 9.696 & 0.329 & 9.718 & 0.557 & 9.281 & 0.236 & 9.586 & 0.329 & 8.785 \\
\hline Taxonomy Parent & 0.626 & 9.102 & 0.553 & 9.165 & 0.419 & 9.319 & 0.685 & 8.658 & 0.319 & 9.171 & 0.431 & 8.515 \\
\hline
\end{tabular}

Table 14. Accuracy on OOD dataset by enforcing class taxonomy: Baseline: <dalmatian>; Stack Parent: <dalmatian, dog, animal>; Taxonomy Parent:<dalmatian, which is type of a dog, which is type of an animal>; Shuffle Parent: <dalmatian, which is type of an organism, which is type of a seabird>; The Taxonomy Parent method, which includes the full hierarchical relationship, yields the best performance, highlighting the effectiveness of incorporating structured knowledge into model predictions.
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline & \multicolumn{2}{|l|}{Element} & \multicolumn{2}{|l|}{ImageNetv2} & \multicolumn{2}{|l|}{ImageNet-S} & \multicolumn{2}{|l|}{ImageNet-R} & \multicolumn{2}{|l|}{ImageNet-A} & \multicolumn{2}{|l|}{ObjectNet} \\
\hline & ID & OOD & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE & KEN & SPE \\
\hline \multirow{4}{*}{ALL} & Top1 & Top1 & 0.840 & 0.947 & 0.170 & 0.092 & 0.146 & 0.042 & 0.068 & 0.037 & 0.317 & 0.339 \\
\hline & LCA & Top1 & 0.421 & 0.517 & 0.779 & 0.923 & 0.761 & 0.911 & 0.730 & 0.888 & 0.867 & 0.967 \\
\hline & Top1 & Top5 & 0.672 & 0.818 & 0.151 & 0.059 & 0.134 & 0.004 & 0.108 & 0.021 & 0.279 & 0.297 \\
\hline & LCA & Top5 & 0.571 & 0.729 & 0.768 & 0.919 & 0.752 & 0.897 & 0.755 & 0.908 & 0.861 & 0.966 \\
\hline \multirow{4}{*}{VLM} & Top1 & Top1 & 0.971 & 0.997 & 0.840 & 0.936 & 0.864 & 0.943 & 0.753 & 0.915 & 0.905 & 0.982 \\
\hline & LCA & Top1 & 0.882 & 0.972 & 0.729 & 0.861 & 0.762 & 0.886 & 0.800 & 0.942 & 0.870 & 0.972 \\
\hline & Top1 & Top5 & 0.908 & 0.980 & 0.848 & 0.951 & 0.882 & 0.959 & 0.753 & 0.910 & 0.842 & 0.964 \\
\hline & LCA & Top5 & 0.900 & 0.981 & 0.746 & 0.879 & 0.775 & 0.907 & 0.794 & 0.943 & 0.829 & 0.955 \\
\hline \multirow{4}{*}{VM} & Top1 & Top1 & 0.948 & 0.993 & 0.771 & 0.901 & 0.743 & 0.887 & 0.735 & 0.877 & 0.822 & 0.927 \\
\hline & LCA & Top1 & 0.910 & 0.981 & 0.740 & 0.882 & 0.705 & 0.862 & 0.741 & 0.851 & 0.790 & 0.918 \\
\hline & Top1 & Top5 & 0.939 & 0.992 & 0.752 & 0.894 & 0.758 & 0.901 & 0.818 & 0.941 & 0.815 & 0.920 \\
\hline & LCA & Top5 & 0.894 & 0.977 & 0.733 & 0.879 & 0.707 & 0.871 & 0.780 & 0.916 & 0.783 & 0.911 \\
\hline
\end{tabular}

Table 15. Ranking measurement of ID LCA/Top1 with OOD Top1/Top5 on 75 models across modality(36 VMs and 39 VLMs); As shown in the 'ALL grouping', LCA shows a much better result in preserving the model relative ranking to model OOD performance on all OOD datasets (with the exception of ImageNet-v2), which indicates its superiority for model selection.