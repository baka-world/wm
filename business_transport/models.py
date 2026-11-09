"""Instrumentation for the existing structured service head (no new network)."""
from lab_transport.service_dynamics import RuleMotionNeuralService


class CountingStructuredService(RuleMotionNeuralService):
    def __init__(self,geometry,network,device,threshold):
        super().__init__(geometry,network,device,threshold)
        self.service_calls=0

    def service_advances(self,features):
        self.service_calls+=1
        return super().service_advances(features)
