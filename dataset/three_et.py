import tonic
import torch
import numpy as np
import os
import warnings

from collections.abc import Generator

PATH = os.path.dirname(__file__)

class ThreeET:
    '''
    Image size: 256x192\n
    Number of data: \n
        Train: 16 * 2000\n
        Test: 2 * 2000\n
    The dataset is not thread-safe, please use it in a single-threaded environment.\n
    '''
    IMAGE_SIZE = (256, 192)
    NUM_TIME_INTERVAL = 2000
    def __init__(self, download: bool = False):
        super().__init__()

        if download:
            warnings.warn(
                "Automatically downloading is provided by tonic. In many cases, it is better to download the dataset manually. See the README for instructions.",
                UserWarning,
                stacklevel=0
            )
        else:
            if not os.path.exists(os.path.join(PATH, "data", "ThreeET_Eyetracking")):
                raise FileNotFoundError(
                    "The dataset ThreeET_Eyetracking is not downloaded. Please download it first. See the README for instructions."
                )
            
            if not os.path.exists(os.path.join(PATH, "data", "ThreeET_Eyetracking", "ThreeET_Eyetracking.zip")):
                with open(os.path.join(PATH, "data", "ThreeET_Eyetracking", "ThreeET_Eyetracking.zip"), "wb") as f:
                    f.write(b'0')
        
        self.raw_data_train = tonic.datasets.ThreeET_Eyetracking(save_to=os.path.join(PATH, "./data"), split='train')
        self.raw_data_test = tonic.datasets.ThreeET_Eyetracking(save_to=os.path.join(PATH, "./data"), split='val')

        self.data_train = (
            torch.zeros((0, self.NUM_TIME_INTERVAL, *self.IMAGE_SIZE), dtype=torch.uint8), 
            torch.zeros((0, self.NUM_TIME_INTERVAL, 2), dtype=torch.float64)
        )
        self.data_test = (
            torch.zeros((0, self.NUM_TIME_INTERVAL, *self.IMAGE_SIZE), dtype=torch.uint8),
            torch.zeros((0, self.NUM_TIME_INTERVAL, 2), dtype=torch.float64)
        )

        self.set_mode('train')
        assert len(self) == 16
        self.set_mode('test')
        assert len(self) == 2
        self.set_mode('train')

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, mode: str, et: "ThreeET"):
                super().__init__()
                self.__mode = mode
                self.__et = et
            
            def __len__(self):
                return self.__et.__len__(self.__mode)
                
            def __getitem__(self, index):
                if index >= len(self):
                    raise IndexError("Index out of range")
                if index < 0:
                    index += len(self)
                if index < 0:
                    raise IndexError("Index out of range")
                return self.__et.__getitem__(index, self.__mode)
            
            def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
                yield self.__et.__iter__(self.__mode)
                
        self.train = Dataset('train', self)
        self.test = Dataset('test', self)

    def __format_events(self, events: np.ndarray) -> torch.Tensor:
        # Format events to the correct type
        t = events['t'].astype(np.uint64) # ascending time stamp
        x = events['x'].astype(np.uint8)
        y = events['y'].astype(np.uint8)
        p = events['p'].astype(np.uint8)
        # Convert to tensor
        time_edges = np.linspace(0, t[-1], self.NUM_TIME_INTERVAL + 1)
        w, h = self.IMAGE_SIZE
        tensor = torch.zeros((self.NUM_TIME_INTERVAL, w, h), dtype=torch.uint8)
        for i in range(self.NUM_TIME_INTERVAL):
            start, end = np.searchsorted(t, time_edges[i: i + 2], side='left')
            if start >= end:
                continue
            slice_x = x[start:end]
            slice_y = y[start:end]
            slice_p = p[start:end]
            mask = slice_p == 1
            if np.any(mask):
                hist, _ = np.histogramdd(
                    (slice_x[mask], slice_y[mask]),
                    bins=(w, h),
                    range=[[0, w], [0, h]]
                )
                tensor[i] += torch.from_numpy(hist.astype(np.uint8))
            mask = slice_p == 0
            if np.any(mask):
                hist, _ = np.histogramdd(
                    (slice_x[mask], slice_y[mask]),
                    bins=(w, h),
                    range=[[0, w], [0, h]]
                )
                tensor[i] -= torch.from_numpy(hist.astype(np.uint8))
        return tensor

    @property
    def train_gen(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        def data():
            for i in range(len(self.raw_data_train)):
                if self.data_train[0].shape[0] <= i:
                    events, targets = self.raw_data_train[i]
                    self.data_train = (
                        torch.cat((self.data_train[0], self.__format_events(events).unsqueeze(0))), 
                        torch.cat((self.data_train[1], torch.from_numpy(targets).unsqueeze(0) / 2.5))
                    )
                yield self.data_train[0][i], self.data_train[1][i]
        return data()
    
    @property
    def test_gen(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        def data():
            for i in range(len(self.raw_data_test)):
                if self.data_test[0].shape[0] <= i:
                    events, targets = self.raw_data_test[i]
                    self.data_test = (
                        torch.cat((self.data_test[0], self.__format_events(events).unsqueeze(0))), 
                        torch.cat((self.data_test[1], torch.from_numpy(targets).unsqueeze(0) / 2.5))
                    )
                yield self.data_test[0][i], self.data_test[1][i]
        return data()

    def set_mode(self, mode: str):
        if mode not in ['train', 'test']:
            raise ValueError("mode must be 'train' or 'test'")
        self.__mode = mode
    
    def __len__(self, mode: str = None) -> int:
        if mode is None:
            mode = self.__mode
        if mode == 'train':
            return len(self.raw_data_train)
        elif mode == 'test':
            return len(self.raw_data_test)
        else:
            raise ValueError("mode must be 'train' or 'test'")
    
    def __getitem__(self, index, mode: str = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mode is None:
            mode = self.__mode
        if index >= len(self):
            raise IndexError("Index out of range")
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("Index out of range")
        if mode == 'train':
            if self.data_train[0].shape[0] <= index:
                for i, item in enumerate(self.train_gen):
                    if i == index:
                        return item
            return self.data_train[index]
        elif mode == 'test':
            if self.data_test[0].shape[0] <= index:
                for i, item in enumerate(self.test_gen):
                    if i == index:
                        return item
            return self.data_test[index]
        else:
            raise ValueError("mode must be 'train' or 'test'")
    
    def __iter__(self, mode: str = None) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        if mode is None:
            mode = self.__mode
        if mode == 'train':
            return self.train_gen
        elif mode == 'test':
            return self.test_gen
        else:
            raise ValueError("mode must be 'train' or 'test'")
        

# dataset = ThreeET()
# print(dataset[3])
