# Recursive function to update nested dictionaries
def update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            if v is not None:
                d[k] = v
    return d


def override_config(config, args_dict):
    overrides = {k: v for k, v in args_dict.items() if v is not None and (k in list(config.keys()))}
    return update(config, overrides)


def new_config(args):
    result = {}
    for k, v in args.items():
        parts = k.split(".")
        temp = result
        for part in parts[:-1]:
            if part not in temp or not isinstance(temp[part], dict):
                temp[part] = {}
            temp = temp[part]
        temp[parts[-1]] = v
    return result


def exp_parser(config, args: dict):
    args_dict = new_config(args)
    updated_config = override_config(config, args_dict)
    return updated_config


def build_exp_suffix(data_config: dict, exp_tag: str = None) -> str:
    """Build experiment save-path suffix from varying params.

    Ensures train_fcst.py and sample_fcst.py generate identical paths.
    """
    suffix = ""
    if data_config.get("label_len", 0) > 0:
        suffix += f"_lw{data_config['label_len']}"
    if exp_tag:
        suffix += f"_{exp_tag}"
    return suffix
