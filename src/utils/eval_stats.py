
from typing import List
from dataclasses import dataclass, field


@dataclass
class EvaluationStats:
    step: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    
    def append_step(self, step: float) -> None:
        self.step.append(step)

    def append_train(self, loss: float) -> None:
        self.train_loss.append(loss)
    
    def append_val(self, loss: float) -> None:
        self.val_loss.append(loss)

    @property
    def last_train_loss(self) -> float:
        return self.train_loss[-1]
    
    @property
    def last_val_loss(self) -> float:
        return self.val_loss[-1]
    
    def __len__(self) -> int:
        return len(self.step)


@dataclass
class EnvEvaluationStats:
    step: List[int] = field(default_factory=list)
    average_return: List[float] = field(default_factory=list)
    
    def append_step(self, step: float) -> None:
        self.step.append(step)

    def append_return(self, avg_return: float) -> None:
        self.average_return.append(avg_return)
    
    @property
    def last_return(self) -> float:
        return self.average_return[-1]
    
    def __len__(self) -> int:
        return len(self.step)