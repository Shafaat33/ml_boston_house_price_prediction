from pydantic import BaseModel
# 2. Class which describes factors of house price predictions
class House_predict(BaseModel):
    CRIM: float 
    ZN: float 
    INDUS: float
    NOX: float 
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float
