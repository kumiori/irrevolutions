import yaml
import hashlib


def parameters_vs_ell(parameters=None, ell=0.1):
    if parameters is None:
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["ell"] = ell
    parameters["geometry"]["ell_lc"] = 3

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def parameters_vs_elle(parameters=None, elle=0.3):
    # for the thin film model
    if parameters is None:
        with open("../data/thinfilm/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["ell_e"] = elle
    parameters["model"]["ell"] = elle / 3
    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["geometry"]["lc"] = (
        parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]
    )
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def parameters_vs_SPA_scaling(parameters=None, s=0.01):
    if parameters is None:
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["stability"]["cone"]["scaling"] = s

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    # parameters["stability"]["cone"]["cone_atol"] = 1e-6
    # parameters["stability"]["cone"]["cone_rtol"] = 1e-5

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def parameters_vs_n_refinement(parameters=None, r=3):
    if parameters is None:
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["geometry"]["ell_lc"] = r

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature
