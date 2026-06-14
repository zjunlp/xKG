## NetHack
When evaluating an agent on NetHack, the agent is rolled out until it dies, 150 steps are taken without progress being made, or 100k steps are taken

To compute the Fisher matrix 10000 batches should be sampled from the NLD-AA dataset.

Existing implementations that should be used:
- The NetHack environment should be implemented using the code at https://github.com/heiner/nle
- APPO NetHack should be implemented using the code at https://github.com/alex-petrenko/sample-factory/
- AutoAscend should be implemented using the code at https://github.com/cdmatters/autoascend/tree/jt-nld
- The existing pre-trained model is the 30M LSTM model from from Tuyls et al. (2023). The weights should be downloaded from https://drive.google.com/uc?id=1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl
- The SAC implementation should implement a similar methodology as described in https://spinningup.openai.com/en/latest/algorithms/sac.html?highlight=SAC

### NLD-AA
NLD-AA can be downloaded following the instructions at https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022. Below we have outlined a summary of downloading and using the dataset:

Start by executing:
```bash
# Download NLD-AA
mkdir -p nld-aa
curl -o nld-aa/nld-aa-dir-aa.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aa.zip
curl -o nld-aa/nld-aa-dir-ab.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ab.zip
curl -o nld-aa/nld-aa-dir-ac.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ac.zip
curl -o nld-aa/nld-aa-dir-ad.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ad.zip
curl -o nld-aa/nld-aa-dir-ae.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ae.zip
curl -o nld-aa/nld-aa-dir-af.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-af.zip
curl -o nld-aa/nld-aa-dir-ag.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ag.zip
curl -o nld-aa/nld-aa-dir-ah.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ah.zip
curl -o nld-aa/nld-aa-dir-ai.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ai.zip
curl -o nld-aa/nld-aa-dir-aj.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-aj.zip
curl -o nld-aa/nld-aa-dir-ak.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ak.zip
curl -o nld-aa/nld-aa-dir-al.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-al.zip
curl -o nld-aa/nld-aa-dir-am.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-am.zip
curl -o nld-aa/nld-aa-dir-an.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-an.zip
curl -o nld-aa/nld-aa-dir-ao.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ao.zip
curl -o nld-aa/nld-aa-dir-ap.zip https://dl.fbaipublicfiles.com/nld/nld-aa/nld-aa-dir-ap.zip
```

Unzip:
```bash
# for NLD-AA
# will give you an nle_data directory at /path/to/dir/nld-aa-dir/nld-aa/nle_data/
unzip /path/to/nld-aa/nld-aa-dir-aa.zip  -d /path/to/dir
unzip /path/to/nld-aa/nld-aa-dir-ab.zip  -d /path/to/dir
unzip /path/to/nld-aa/nld-aa-dir-ac.zip  -d /path/to/dir
...
```

The code needed to use the dataset can be found on the main branch of [NLE](https://github.com/facebookresearch/nle). You can follow the instructions to install there, or try the below.
```bash
# With pip:
pip install git+https://github.com/facebookresearch/nle.git@main

# From source:
git clone --recursive https://github.com/facebookresearch/nle.git
cd nle && pip install -e .
```

Once this is installed, you simply need to load the nld folders (once) which will create a small local sqlite3 database, and then you can use the dataset.
```python
import nle.dataset as nld

if not nld.db.exists():
    nld.db.create()
    # NB: Different methods are used for data based on NLE and data from NAO.
    nld.add_nledata_directory("/path/to/nld-aa", "nld-aa-v0")
    nld.add_altorg_directory("/path/to/nld-nao", "nld-nao-v0")

dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=128, ...)
for i, mb in enumerate(dataset):
    foo(mb) # etc...
```

## Montezuma's Revenge
To clarify how models are trained: 
1. You first pre-train a model (I'll call this M1) - a PPO agent with RND until it achieved an episode cumulative reward of around 7000
2. You collect 500 trajectories from M1 that start from room 7 onwards
3. You use the 500 trajectories to train M2 - this is the "pre-trained" model
4. M2 is used for fine-tuning other models
5. The "from scratch" model doesn't use any behavioral cloning

Existing implementations that should be used:
- PPO RND with the environment should be implemented using the code at https://github.com/jcwleo/random-network-distillation-pytorch

Figure 4 is not required to be reproduced

In Figure 6, the success rate should be computed every 5 million training steps.

## Robotic Sequence
In Figure 8, the log-likelihoods should be computed every 50 thousand training steps.

The "pre-training" model is trained on scratch on all stages.
