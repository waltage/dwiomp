from enum import Enum

class VarType(Enum):
    LogSpace = 1
    UnitSpace = 2


x = VarType.LogSpace
print(x)
