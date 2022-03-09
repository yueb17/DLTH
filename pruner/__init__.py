from . import reg_pruner, l1_pruner
from . import l1_pruner_iterative, reg_pruner_iterative

pruner_dict = {
    'RST': reg_pruner,
    'L1': l1_pruner,
    'L1_Iter': l1_pruner_iterative,
    'RST_Iter': reg_pruner_iterative
}