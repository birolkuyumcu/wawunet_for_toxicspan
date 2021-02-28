# wawUnet for SemEval-2021 Task 5: Toxic Spans Detection

## SemEval 2021 Task 5: Toxic Spans Detection

Moderation is crucial to promoting healthy online discussions. Although several toxicity (abusive language) detection datasets and models have been released, most of them classify whole comments or documents, and do not identify the spans that make a text toxic. But highlighting such toxic spans can assist human moderators (e.g., news portals moderators) who often deal with lengthy comments, and who prefer attribution instead of just a system-generated unexplained toxicity score per post. The evaluation of systems that could accurately locate toxic spans within a text is thus a crucial step towards successful semi-automated moderation.

## wawUnet
Our approach considers toxic spans detection as a segmentation problem. The system, Waw-unet, consists of a 1-D convolutional neural network adopted from U-Net architecture commonly applied for semantic segmentation. We customize existing architecture by adding a special network block considering for text segmentation, as an essential
component of the model. 

read details of wawunet at <a href="wawunet.md" target="_blank">wawunet.md</a>

## Papers with wawUnet

1. <a href="https://ieeexplore.ieee.org/abstract/document/9302154" target="_blank">Parsing Address Texts with Deep Learning Method</a>




