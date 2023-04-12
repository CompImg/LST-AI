# lst.ai

ToDo:
-----
> Model diagnostics / evaluation pipeline

> SAMSeg

> nnUNet (5fold)

Paper:
------
> https://www.nature.com/articles/s41598-023-31207-5

> https://www.nature.com/articles/s41598-020-79925-4

> https://arxiv.org/abs/2005.12209

Ideen für Simultaneous Image Synthesis and SegmentatIon:
--------------------------------------------------------
> Use [attention loss](https://pubmed.ncbi.nlm.nih.gov/35557607/) for synthesis? => Does the segmentation mask come from the gt or network?

> How to weigh the loss terms (Dice/BCE vs. SSIM)?

> Include dropout?

> Have the discriminator also look at the segmentation?

> "W-Net" architecture -> First synthesize DIR, then segment it? Shared weights?
