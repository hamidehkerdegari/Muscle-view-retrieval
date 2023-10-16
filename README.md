# Automatic Retrieval of Corresponding US Views in Longitudinal Examinations
This repository is an official implementation of our MICCAI2023 paper [Automatic Retrieval of Corresponding US Views in Longitudinal Examinations](https://arxiv.org/abs/2306.04739).

![Modelarch](https://github.com/hamidehkerdegari/Muscle-view-retrieval/assets/30697849/d3b53948-9b6b-46fc-9d96-755c3b2b7fb5)

## Abstact
Skeletal muscle atrophy is a common occurrence in critically ill patients in the intensive care unit (ICU) who spend long periods in
bed. Muscle mass must be recovered through physiotherapy before patient discharge and ultrasound imaging is frequently used to assess the
recovery process by measuring the muscle size over time. However, these manual measurements are subject to large variability, particularly since
the scans are typically acquired on different days and potentially by different operators. In this paper, we propose a self-supervised contrastive learning approach to automatically retrieve similar ultrasound muscle views at different scan times. Three different models were compared using data from 67 patients acquired in the ICU. Results indicate that our contrastive model outperformed a supervised baseline model in the task of view retrieval with an AUC of 73.52% and when combined with an automatic segmentation model achieved 5.7% ± 0.24% error in cross-sectional area. Furthermore, a user study survey confirmed the efficacy of our model for muscle view retrieval.

## Dependencies

* Linux, CUDA>=11.2
  
* Python>=3.8
  
We recommend you to create a virtual environment as follows:

> python -m venv virtual-environment-name 

Then, activate the environment:

> source virtual-environment-name/bin/activate

Other requirements are:

> pip install -r requirements.txt


## Dataset preparation
All the required codes for dataset preparation is inside "/utils" folder.

## Pretrained model
The path to the pretrained model "model/simclr.py" and the path to the pretrained weight "checkpoints/ss"

## Downstream task
The path to the downstream task (classification task) "/main_ssl_eval_simclr.py"



**Authors:**

* Hamideh Kerdegari
* Nhat Phung
* Alberto Gomez


## Citation

If you find this paper useful in your research, please consider citing:
 
```bibtex
@article{kerdegari2023automatic,
  title={Automatic retrieval of corresponding US views in longitudinal examinations},
  author={Kerdegari, Hamideh and Phung, Tran Huy Nhat and Nguyen, Van Hao and Truong, Thi Phuong Thao and Le, Ngoc Minh Thu and Le, Thanh Phuong and Le, Thi Mai Thao and Pisani, Luigi and Denehy, Linda and Consortium, Vital and others},
  journal={arXiv preprint arXiv:2306.04739},
  year={2023}
}
```

## Contact information
If you have any questinos, please contact me via hamideh.kerdegari@gmail.com

# Acknowledgement
This work was supported by the Wellcome Trust UK (110179/Z/15/Z, 203905/Z/16/Z, WT203148/Z/16/Z), by the Department of Health via the National Institute for Health Research (NIHR) comprehensive Biomedical Research Centre at Guy's and St Thomas' NHS Foundation Trust in partnership with King's College London and King's College Hospital NHS Foundation Trust. The views expressed are those of the author(s) and not necessarily those of the NHS, the NIHR or the Department of Health.
