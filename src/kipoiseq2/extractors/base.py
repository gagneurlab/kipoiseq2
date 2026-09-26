import abc
import logging

from kipoiseq2 import Interval

log = logging.getLogger(__name__)

__all__ = ["BaseExtractor"]


class BaseExtractor:
    __metaclass__ = abc.ABCMeta

    _use_strand: bool

    # main method
    @abc.abstractmethod
    def extract(self, interval: Interval, *args, **kwargs) -> str:
        raise NotImplementedError

    @property
    def use_strand(self):
        return self._use_strand

    # closing files
    def __del__(self):
        return self.close()

    def close(self):
        # implemented by the subclass
        pass
