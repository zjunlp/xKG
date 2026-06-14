# Clarifications

- In Section 3.1 Frequency-Threshold based Forcasting, the authors meant $\hat{D}_{PT}$ (the pre-training dataset, filtering out samples the model got wrong) instead of $D_{PT}$ (the entire pre-training dataset). All forecasting algorithms in Table 1 and Table 2 are evaluated on $\hat{D}_{PT}$.

- In Section 4.1 Training and Evaluation Setup, the paper describes how $D_{PT}$ is constructed, saying that they "evaluate forgetting over 36 tasks from the training split of the Public Pool of Prompts (P3) dataset" and "use a balanced sample of 100 examples per task to from our $D_{PT}$". By balanced, the paper simply means they sampled an equal number (100) of examples per task. The 36 tasks used are the intersection of T0-Train tasks (Table 5 in https://arxiv.org/pdf/2110.08207) and the tasks in used in the repo of BART0 model (https://github.com/INK-USC/ReCross/blob/main/data/):
    - glue-mrpc
    - glue-qqp
    - paws_x-en
    - kilt_tasks-hotpotqa
    - wiki_qa
    - adversarial_qa-dbert
    - adversarial_qa-dbidaf
    - adversarial_qa-droberta
    - duorc-SelfRC
    - duorc-ParaphraseRC
    - ropes
    - quoref
    - cos_e-v1.11
    - cosmos_qa
    - dream
    - qasc
    - quail
    - quartz
    - sciq
    - social_i_qa
    - wiki_hop-original
    - wiqa
    - amazon_polarity
    - app_reviews
    - imdb
    - rotten_tomatoes
    - yelp_review_full
    - common_gen
    - wiki_bio
    - cnn_dailymail-3.0.0
    - gigaword
    - multi_news
    - samsum
    - xsum
    - ag_news
    - dbpedia_14

- In Section 4.1 Tasks for Model Refinement, the paper says that MMLU is used to refine FLAN-T5, however the specific split isn't specified. The validation set is used, specifically the one from the original release of MMLU (https://people.eecs.berkeley.edu/~hendrycks/data.tar).

- In Section 4.1 Training and Evaluation of the Forecasting Model, the paper says that they "collect mis-predicted examples from the training split $D^{Train}_R$ and the test split $D^{Test}_R$", however the process by which they're collected isn't specified. First, $D_R$ is constructed by evaluating the pretrained LMs on each new task and keeping only those examples where model predictions are wrong. Here, a prediction is graded using the exact match metric using the evaluation script of SQuAD 2.0 (https://rajpurkar.github.io/SQuAD-explorer/). Second, $D_R$ is randomly split into 60% and 40% subsets to create $D^{Train}_R$ and $D^{Test}_R$ respectively. More details about $D_R$: For T5 experiments, validation split of 57 tasks from MMLU was used. For BART0 experiments, the test split in https://github.com/INK-USC/ReCross/blob/main/data/ was used, and the set of test tasks used were:
    - super_glue-wsc.fixed
    - winogrande-winogrande_xl
    - super_glue-cb
    - super_glue-rte
    - anli
    - super_glue-copa
    - hellaswag
    - super_glue-wic

- There are no specifics as for how LoRA was applied to FLAN-T5_{Large} in Section 4. LoRA was applied to the query and value matrices in all self-attention layers:

    ```python
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="none", 
        target_modules=['q', 'v'],
    )
    ```

- In Appendix B, the paper says that the P3 dataset has been split into in- and out-of-distribution splits by task (e.g. `anli` is in the in-distribution split). There are many variants of each task in the P3 dataset, though (e.g. for the `anli` task, there's `anli_claim_true_false_inconclusive_r1_score_eval`, `anli_claim_true_false_inconclusive_r1`, etc) but it's not specified whether all variants are used, or just a subset. All task variants are used.

- The full fine-tuning setup was used to produce Table 2, **not** head-only or LoRA.

- The hyperparameters specified in Section 4.1 -- Hyperparameters were used to produce Figure 3.

- When fine-tuning models to produce Figure 3, each example $(x_i, y_i) \in D_R^{test}$ is sampled at random. That is, the examples in each data stream were randomly shuffled.

- In Figure 3, is the forecasted forgetting binary indicator computed by $g$ at the start and used for every time step (i.e. $\hat{z}_{ij}$) or is it computed each timestep (i.e. $\hat{z}_{ij}^{t}$)? The forgetting is computed at the start and used for every time step.


# Out of Scope

- In Table 3, results from other papers are mentioned (MIR and OCS); these are out of scope.

- The "Hyperparameter Analysis" sub-section in Section 5.2, Section 5.3 and Table 5 are all out of scope.
