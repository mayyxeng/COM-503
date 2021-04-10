
from enum import Enum


class Job:
  class Type(Enum):
    TYPE1 = 1
    TYPE2 = 2
  def __init__(self, _type):
    self.type = _type
  