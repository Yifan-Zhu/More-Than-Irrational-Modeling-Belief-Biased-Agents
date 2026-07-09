from abc import ABC

class BasePipeline(ABC):
    def __init__(self, theta_true=None, agent_params=None, agent_class=None, crci_class=None, crci_params=None):
        self.theta_true = theta_true
        self.agent_params = agent_params
        self.agent_class = agent_class
        self.crci_class = crci_class
        self.crci_params = crci_params
        if agent_params is not None:
            self.agent = agent_class(**agent_params)
        if crci_params is not None:
            self.crci = crci_class(**crci_params)
        self.posterior = None

