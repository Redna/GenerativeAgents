from pydantic import BaseModel, root_validator
from typing import Dict, List
import networkx as nx

class LocationConfig(BaseModel):
    name: str
    description: str
    locations: Dict[str, 'LocationConfig'] = None
    connections: List[Dict[str, str]] = []
    
    @root_validator
    def check_connects_with(cls, values):
        if not values.get("locations") or not values.get("connections"):
            return values
        
        location_names = [name for name, _ in values.get("locations").items()]
        
        for connection in values.get("connections"):
            wrong_connection = [f"{f} <-> {t}" for f,t in connection.items() if f not in location_names or t not in location_names]
            
            if wrong_connection:
                raise ValueError(f"The connection <{wrong_connection}> does not exist in {location_names}")
        
        return values
    
class SimulationConfig(BaseModel):
    world: LocationConfig
