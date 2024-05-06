
from pydantic import BaseModel


class ActionEvent(BaseModel):
    subject: str
    predicate: str
    object: str