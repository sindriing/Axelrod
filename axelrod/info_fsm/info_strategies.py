# Test file
import axelrod as axl
from axelrod.info_fsm.info_fsm import InfoFSM
from axelrod.action import Action

C,D = Action.C, Action.D

# Formulas for FSM
TFT_formula = ((1,2,0,C),
               (1,2,0,C),
               (0,2,0,D))

blinkingTFT_formula = ((1,2,0,C),
                       (1,2,1,C),
                       (0,2,0,D))
dblblinkingTFT_formula = ((1,2,0,C),
                         (1,2,1,C),
                         (0,2,1,D))

alternator_formula = ((1,1,0,C),
                      (0,0,0,D))

double_alternator_formula = ((1,1,1,C),
                             (0,0,1,D))

cooperator_formula = ((0,0,0,C),)
defector_formula = ((0,0,0,D),)

InfoTFT = InfoFSM(TFT_formula)
InfoTFT2 = InfoFSM(blinkingTFT_formula)
InfoTFT3 = InfoFSM(dblblinkingTFT_formula)
InfoAlternator = InfoFSM(alternator_formula)
InfoDoubleAlternator = InfoFSM(double_alternator_formula)
cooperator = InfoFSM(cooperator_formula)
defector = InfoFSM(defector_formula)
