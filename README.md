# From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking

[![Arxiv](https://img.shields.io/badge/ArXiv-2512.02392-B31B1B.svg)](https://arxiv.org/abs/2512.02392)


> **TL;DR.** We reveal that DETR-based end-to-end MOT suffers from overly similar object embeddings. FDTA explicitly enhances discriminativeness in this paradigm.

![Teaser](./assets/teaser.png)

## 📢 News
* **[Coming Soon]** The code and dataset are being organized and will be released shortly. Please star this repo for updates!

## Main Results

### DanceTrack

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---------------|------|------|------|------|------|
| train         | 71.7 | 77.2 | 63.5 | 91.3 | 81.0 |
| train+val     | 74.4 | 80.0 | 67.0 | 92.2 | 82.7 |

### SportsMOT

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---------------|------|------|------|------|------|
| train         | 74.2 | 78.5 | 65.5 | 93.0 | 84.1 |

### BFT

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---------------|------|------|------|------|------|
| train         | 72.2 | 84.2 | 74.5 | 78.2 | 70.1 |

## Acknowledgements

The code is built on top of these awesome repositories. We thank the authors for opensourcing their code.

* [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
* [MOTR](https://github.com/megvii-research/MOTR)
* [MOTIP](https://github.com/MCG-NJU/MOTIP)
* [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR)
* [TrackEval](https://github.com/JonathonLuiten/TrackEval)


##  Citation
If you find our work useful for your research, please consider citing:

```bibtex
@article{shao2025fdta,
  title={From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking},
  author={Shao, Yuqing and Yang, Yuchen and Yu, Rui and Li, Weilong and Guo, Xu and Yan, Huaicheng and Wang, Wei and Sun, Xiao},
  journal={arXiv preprint arXiv:2512.02392},
  year={2025}
}