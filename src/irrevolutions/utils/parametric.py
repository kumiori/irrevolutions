import hashlib
import yaml
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files

def parameters_vs_ell(parameters=None, ell=0.1):
    """
    Update the model parameters for a given value of 'ell'.

    This function modifies the 'ell' parameter. 
    If no parameters are provided, it loads them from the default file.

    Args:
        parameters (dict, optional): Dictionary of parameters. 
                                      If None, load from "../test/parameters.yml".
        ell (float, optional): The new 'ell' value to set in the parameters. 
                               Default is 0.1.

    Returns:
        tuple: A tuple containing the updated parameters dictionary and 
               a unique hash (signature) based on the updated parameters.
    """
    if parameters is None:    
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)
        
    parameters["model"]["ell"] = ell

    # Generate a unique signature using MD5 hash based on the updated parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature


def parameters_vs_elle(parameters=None, elle=0.3):
    """
    Update the model parameters for a given value of 'elle' in the thin film model.
    ell_e is the elastic lengthscale, depending on the film and the substrate properties.
    
    Args:
        parameters (dict, optional): Dictionary of parameters. 
                                      If None, load from "../data/thinfilm/parameters.yml".
        elle (float, optional): The new 'ell_e' value to set in the parameters. 
                                Default is 0.3.

    Returns:
        tuple: A tuple containing the updated parameters dictionary and 
               a unique hash (signature) based on the updated parameters.
    """
    if parameters is None:    
        with open("../data/thinfilm/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["ell_e"] = elle
    # parameters["model"]["ell"] = elle / 3

    # Generate a unique signature using MD5 hash based on the updated parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature


def parameters_vs_SPA_scaling(parameters=None, s=0.01):
    """
    Update the stability scaling parameter for the SPA (Second-Order Analysis).

    This function updates the scaling factor in the cone algorithm for the stability analysis.

    Args:
        parameters (dict, optional): Dictionary of parameters. 
                                      If None, load from "../test/parameters.yml".
        s (float, optional): The new scaling value to set in the parameters. 
                             Default is 0.01.

    Returns:
        tuple: A tuple containing the updated parameters dictionary and 
               a unique hash (signature) based on the updated parameters.
    """
    if parameters is None:    
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["stability"]["cone"]["scaling"] = s

    # Generate a unique signature using MD5 hash based on the updated parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature


def parameters_vs_n_refinement(parameters=None, r=3):
    """
    Update the refinement parameter for the geometry mesh resolution.

    This function modifies the mesh resolution parameter 'ell_lc' 
    in the geometry section of the parameters. 

    Args:
        parameters (dict, optional): Dictionary of parameters. 
                                      If None, load from "../test/parameters.yml".
        r (int, optional): The new refinement value to set in the parameters. 
                           Default is 3.

    Returns:
        tuple: A tuple containing the updated parameters dictionary and 
               a unique hash (signature) based on the updated parameters.
    """
    if parameters is None:    
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["geometry"]["ell_lc"] = r

    # Generate a unique signature using MD5 hash based on the updated parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature


def update_parameters(parameters, key, value):
    """
    Recursively traverses the dictionary d to find and update the key's value.
    
    Args:
    d (dict): The dictionary to traverse.
    key (str): The key to find and update.
    value: The new value to set for the key.
    
    Returns:
    bool: True if the key was found and updated, False otherwise.
    """
    if key in parameters:
        parameters[key] = value
        return True

    for k, v in parameters.items():
        if isinstance(v, dict):
            if update_parameters(v, key, value):
                return True
    
    return False
