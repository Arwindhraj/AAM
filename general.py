import os
import json

class Config():
    """Loads configuration of the model into class variables"""
    def __init__(self):
        with open('config.json','r') as config_file:
            data = config_file.read()
            self.data=json.loads(data)
        for key,value in self.data.items():
            setattr(self,key,value) 

       
        

