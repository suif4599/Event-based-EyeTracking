from abc import ABC, abstractmethod
import torch


class NoiseBase(ABC):
    '''
    Base class for noise datasets.\n
    The dataset is not thread-safe, please use it in a single-threaded environment.\n
    '''
    @abstractmethod
    def apply(self, image_stack: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        '''
        Apply noise to the image stack.\n
        Args:\n
            image_stack (torch.Tensor): Image stack of shape (N, W, H)\n
        Returns:\n
            torch.Tensor: Noisy image stack of shape (N, W, H)\n
        '''
        pass
