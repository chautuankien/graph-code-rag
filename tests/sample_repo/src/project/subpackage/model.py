from src.project.base import BaseModel

class ModelTest(BaseModel):
    def get_name(self) -> str:
        return "test"
