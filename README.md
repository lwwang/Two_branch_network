# Two-Branch Neural Networks

## Usage:
* Train a model from scratch: Set the path to the training dataset and checkpoint save directory in run_two_branch.sh. Run sh run_two_branch.sh --train.
* Evaluate a model on the validation set while training: Set the path to validation dataset and checkpoint save directory. Additionally, adjust the time interval between each evaluation in eval_embedding.py. By default, the script will evaluate once per minute on the newest checkpoint in the given checkpoint directory. Run sh run_two_branch.sh --val.
* Evaluate a model on a specific checkpoint: Set the path to the test dataset and checkpoint MetaGraph (.meta file). Run sh run_two_branch.sh --test.
* Use a pre-trained model: Download checkpoints from the URLs below. Follow the instruction for evaluating model on a specific checkpoint.

## Dataset:
**Due to the size of the features (~17G for MSCOCO and 7G for Flickr30K), only the test split is available for download.**
* [Flickr30K Test Split](https://drive.google.com/open?id=12wu0_S8j5tKSSrNHkm_nmy-NSlFhl4iz)
* [MSCOCO Test Split](https://drive.google.com/open?id=11HvzcK_0EyP5JTth_AwCFgISNTzKX5PR)

## Pre-trained models:
* [Flickr30K checkpoint](https://drive.google.com/open?id=1oSOFU73zm6gzx3VZEszq2athX5b4NREV)
* [MSCOCO checkpoint](https://drive.google.com/open?id=1HTXoHsnhj5oRH4c-z60rYw67Els6gIpV)

##If you find our code helpful, please cite our Two-Branch Network Papers:
*@inproceedings{wang2016learning,
  title={Learning deep structure-preserving image-text embeddings},
  author={Wang, Liwei and Li, Yin and Lazebnik, Svetlana},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5005--5013},
  year={2016}
}
*@article{wang2017learning,
  title={Learning Two-Branch Neural Networks for Image-Text Matching Tasks},
  author={Wang, Liwei and Li, Yin and Lazebnik, Svetlana},
  journal={arXiv preprint arXiv:1704.03470},
  year={2017}
}
