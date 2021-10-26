This repo stores the official code for the paper

```
@inproceedings{NEURIPS2019_00989c20,
 author = {Dang, Trung and Thakkar, Om and Ramaswamy, Swaroop and Mathews, Rajiv and Chin, Peter and Beaufays, Françoise},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Revealing and Protecting Labels in Distributed Training},
 url = {},
 year = {2021}
}
```

# Instructions

## Reveal image labels

```
cd image_recognition
python export_gradients.py --model ResNet50 --batch_size 10
python rlg.py --model ResNet50 --batch_size 10
```

## Reveal speech transcript

This will be uploaded soon.