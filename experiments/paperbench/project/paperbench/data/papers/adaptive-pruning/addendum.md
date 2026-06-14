## Experiments
The speed of inference is measured as the inference throughput (sampled processed per second).

The "train time" is measured as the time-to-accuracy of reaching 97% of the dev (/test) set performance of the finetuning baseline

The "relative accuracy" in Seciton 5.5 is the average of the SST2 and MNLI scores of the trained model relative to the average of the accuracy of the SST2 and MNLI scores of the finetuned baseline.

The the speed of inference during evaluation is measured as the inference throughput (sampled processed per second)

All results involving LLaMa models are not required for replication (but results involving other models are still required). As such, the evaluations in the lm-eval-harness, and the Alpaca dataset, are also not required for replication.

The experiments involved in this text from the paper are not required for replication.
> Our analysis in Appendix H demonstrates that controlled adaptive tuning with early pruning during fine-tuning improves LM end-task accuracy better with less training time and memory costs.

The experiments involved in this text from the paper are also not required for replication.
> Detailed comparisons of self-distillation and traditional, static distillation strategies are shown in Appendix G.

## Baselines
The mask tuning baseline should be implemented using https://github.com/WoosukKwon/retraining-free-pruning. Note, you may need to adapt it so it can be applied to a LoRA-tuned model.

The CoFi baseline should be implemented using https://github.com/princeton-nlp/CoFiPruning. Note, you may need to adapt it to make sure only LoRA and L0 modules are tuned.

The default hyperparameters from these repos should be used, unless otherwise stated.

## APT Implementation
During training, the outlier-aware salience of each block is computed as an exponential moving-average $\overline{S}^{(t)}(m) \gets 0.85 \overline{S}^{(t-1)}(m) + 0.15 \hat{S}(m)$, where $\overline{S}^{(t)}(m)$ is the moving-average of block $m$ at time step $t$, and $\hat{S}(m)$ is the current outlier-aware salience score of block $m$

$\mu$ introduced in equation 7 is set to 0 before the pruning starts, and then linearly increased as the pruning goes on while finally reaching 1 at the end of the pruning. Below is an example implementation:
```python
mu = min(1., (global_step - pruning_start_step) / (pruning_end_step - pruning_start_step))
```

The teacher-student layer-mapping is re-computed every training step.

For classification (GLUE) tasks, the distillation loss is calculated as $\mathcal{L}_{\text{distill}} = \mathcal{L}_{\text{pred}} + 0.9 \mathcal{L}_{\text{layer}}$, and for SQuAD and CNN/DM, we have $\mathcal{L}_{\text{distill}} = 0.1 \mathcal{L}_{\text{pred}} + 0.9 \mathcal{L}_{\text{layer}}$

To measure the max GPU memory utilization, use `torch.cuda.max_memory_allocated()` (or equivalent if using a different framework).

$\tau$ in equation 7 is set to 4, following the original CoFi paper.
