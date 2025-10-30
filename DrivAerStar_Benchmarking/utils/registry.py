from pytorch_lightning.loggers import CSVLogger
import inspect


class_registry = {
    'CSVLogger':CSVLogger,
}

def register_class(name=None, ):
    """
    A decorator to register a class in the class_registry with optional additional keys.

    Args:
        name (str or list of str, optional): The name(s) to be used as the key(s) in the registry.
                                             Defaults to cls.__name__ if not provided.

    Returns:
        function: The decorator function that registers the class.
    """
    def decorator(cls):
        if name is None:
            names = [cls.__name__]
        elif isinstance(name, str):
            names = [cls.__name__, name]
        else:
            names = name + [cls.__name__]

        for key in names:
            class_registry[key] = cls

        return cls

    return decorator


def class_builder(config, **kw):
    """
    Creates an instance of a class based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration for the class.
            - 'type' (str): The type of the class to be instantiated. Must be a key in class_registry.
            - 'args' (dict): A dictionary of arguments to be passed to the class constructor.
        **kw: Additional keyword arguments to be passed to the class constructor.

    Returns:
        object: An instance of the class specified in the configuration.

    Raises:
        AssertionError: If 'type' or 'args' are not in the config or if 'type' is not registered in class_registry.
        
    Example usage:
        Assuming SomeClass is defined and registered in class_registry
        config = {
            'type': 'SomeClass',
            'args': {
                'arg1': value1,
                'arg2': value2
            }
        }
        instance = class_builder(config, use_wandb=True)


    """
    assert 'type' in config, "The configuration must contain a 'type' key."
    assert 'args' in config, "The configuration must contain an 'args' key."
    # print(class_registry.keys())
    assert config['type'] in class_registry, f"class {config['type']} not registered in class_registry ({class_registry.keys()})"

    CLASS_NAME = class_registry[config['type']]
    # Get the signature of the __init__ method
    init_signature = inspect.signature(CLASS_NAME.__init__)

    for k in ['use_wandb', 'normalizer_y', 'normalizer_x' , 'save_dir', 'point_normalizer_x', 'point_normalizer_y']:
        if k in init_signature.parameters:
            if k in kw:
                config['args'][k] = kw[k]
            else:
                print(f"{k} is not provided by class_builder, but {config['type']} requires it.")
            
    return CLASS_NAME(**config['args'])


def class_name(config, **kw):
    CLASS_NAME = class_registry[config['type']]
   
            
    return CLASS_NAME 

