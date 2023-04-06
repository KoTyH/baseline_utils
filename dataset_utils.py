import cv2
import glob
import torch
import numpy as np
from PIL import Image
import tensorflow as tf
from torch.utils.data import Dataset
from typing import Tuple, Callable, Any, List, Union
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


# Tensorflow dataset
class TensorflowDataset(tf.keras.utils.Sequence):
    def __init__(self, data: List[str], _transforms: Tuple[Callable[[Any], Any], ...], _augs, data_type: str = 'image'):
        self.data = data
        self._transforms = _transforms
        self._augs = _augs
        self.data_type = data_type

    def __process(self, image: np.ndarray) -> Union[tf.Tensor, np.ndarray]:
        if self._augs:
            aug_input = {"image": image}
            image = self._augs(**aug_input)['image']
        if self._transforms:
            image = self.__apply_transforms(image.astype(np.uint8))
        return image

    # метод для применения трансформов к изображению
    def __apply_transforms(self, image: np.ndarray) -> Union[tf.Tensor, np.ndarray]:
        for trans in self._transforms:
            image = trans(image)
        return image

    # метод для чтения изображений с диска в RGB формате
    def __get_image(self, idx: int) -> np.ndarray:
        return cv2.imread(self.data[idx])[..., ::-1]  # RGB image

    def __getitem__(self, idx: int):
        if self.data_type == 'image':
            return self.__process(self.__get_image(idx))
        else:
            return self.__video_process(idx)

    def __len__(self):
        return len(self.data)


# Pytorch dataset
class PytorchDataset(Dataset):
    def __init__(self, data: List[str], _transforms, _augs, data_type: str = 'image'):
        self.data = data
        self._transforms = _transforms
        self._augs = _augs
        self.data_type = data_type

    def __process(self, image: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        if self._augs:
            aug_input = {"image": image}
            image = self._augs(**aug_input)['image']
        if self._transforms:
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            image = self._transforms(image)
        return image

    # метод для чтения изображений с диска в RGB формате
    def __get_image(self, idx: int) -> np.ndarray:
        return cv2.imread(self.data[idx])[..., ::-1]  # RGB image

    def __getitem__(self, idx: int):
        if self.data_type == 'image':
            return self.__process(self.__get_image(idx))
        else:
            return self.__video_process(idx)

    def __len__(self):
        return len(self.data)


# класс для получения датасета
class ImageDataset:
    def __init__(self, dataset_directory: str, formats, framework_mode: str, data_type_mode: str, _transforms=None, _augs=None, 
                 train: bool = False, split: float = 0.8):
        self.dataset_directory = dataset_directory  # путь до папки с нужными данными
        self.formats = formats  # формат изображений
        self.framework_mode = framework_mode  # pytorch / tensorflow
        self.data_type_mode = data_type_mode  # image / video
        self._transforms = _transforms  # трансформы для подачи в сеть
        self._augs = _augs  # аугментации изображения

        self.full_data = self.__get_dataset_files()

        len_train = int(len(self.full_data) * split)
        len_val = len(self.full_data) - len_train

        if train:
            self.dataset = self.full_data[:len_train]
        else:
            self.dataset = self.full_data[len_val:]

    # метод для получения всех файлов из датасета
    def __get_dataset_files(self) -> List[str]:
        return [p for f in self.formats for p in glob.glob(f"{self.dataset_directory}/*.{f}")]

    # метод получения датасета в соответствие с выбраным фреймворком и типом файлов
    def get_dataset(self):
        if self.framework_mode == 'pytorch':
            return PytorchDataset(self.dataset, self._transforms, self._augs, data_type=self.data_type_mode)
        elif self.framework_mode == 'tensorflow':
            return TensorflowDataset(self.dataset, self._transforms, self._augs, data_type=self.data_type_mode)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    
    _dataset_directory = 'C:/Users/Ilya/Pictures/1'
    _file_formats = ['png', 'jpg', 'jpeg']
    _framework_mode = 'tensorflow'
    _data_type_mode = 'image'

    if _framework_mode == 'pytorch':
        _transforms = Compose([Resize((224, 224)), ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        _transforms = (lambda x: tf.image.resize(x, (224, 224)),
                       lambda x: tf.keras.applications.resnet50.preprocess_input(x),
                       lambda x: tf.reshape(x, [1, 3, 224, 224]))
    
    reader = ImageDataset(_dataset_directory, _file_formats, _framework_mode, _data_type_mode, _transforms=_transforms, train=True)
    result = reader.get_dataset()
    
    for i in result:
        print('result = ', i.shape, type(i))
