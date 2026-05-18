import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import json
import os

def build_net(i_max_ka=1.2, vmin=0.95, vmax=1.05):
    net = pn.case39()
    # Voltage bounds
    net.bus["min_vm_pu"] = vmin
    net.bus["max_vm_pu"] = vmax
    # Thermal limits for lines and transformers
    net.line["max_i_ka"] = i_max_ka
    # Case 39 often has trafos; ensure they have loading limits
    net.trafo["max_loading_percent"] = 100.0 
    return net

def apply_total_load(net, total_p_mw, pf=0.95):
    """Scales load based on total P and a constant Power Factor."""
    base_p = net.load["p_mw"].values
    base_total = base_p.sum()
    k = float(total_p_mw) / float(base_total)
    net.load["p_mw"] = base_p * k
    # Q = P * tan(acos(pf))
    q_ratio = np.tan(np.arccos(pf))
    net.load["q_mvar"] = net.load["p_mw"] * q_ratio

class Model:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.results = None
        # Trigger the load immediately
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """
        Loads the grid configuration from the JSON file created by the API.
        """
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                config = json.load(f)
            
            # Rebuild the network with the user-specified parameters
            self.model = build_net(
                i_max_ka=config.get("i_max_ka", 1.2),
                vmin=config.get("vmin", 0.95),
                vmax=config.get("vmax", 1.05)
            )
        else:
            # Fallback to defaults if file is missing
            self.model = build_net()
    
    def predict(self, loads):
        """
        Runs power flow for each row in the dataframe.
        Expects df to have a column representing total load (MW).
        """
        # Giskard/Scanners usually pass a DataFrame; we extract the load column
        total_p_mws = loads
        detailed_results = []
        
        for total_p_mw in total_p_mws:
            apply_total_load(self.model, total_p_mw)
            
            try:
                pp.runpp(self.model, init="auto")
                
                detailed_results.append({
                    'over_line': self.model.res_line["loading_percent"].to_numpy(),
                    'over_trans': self.model.res_trafo["loading_percent"].to_numpy(),
                    'vm': self.model.res_bus["vm_pu"].to_numpy()
                })
            
            except pp.LoadflowNotConverged:
                detailed_results.append(None)
        
        self.results = detailed_results
        return np.ones(len(detailed_results))
        
        