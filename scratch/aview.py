'''
Created on 15.03.2016

@author: mkennert
'''
from abc import abstractmethod


class AView(object):
        
    @abstractmethod
    def set_height(self,value):
        pass
    
    @abstractmethod
    def set_width(self, value):
        pass
    
    @abstractmethod
    def set_percent(self, value):
        pass
    
    @abstractmethod
    def add_layer(self, percent,name):
        pass
    
    @abstractmethod
    def delete_layer(self):
        pass
    
    @abstractmethod
    def update_layer_information(self,name,price,density,stiffness,strength,percent):
        pass
    
        
        