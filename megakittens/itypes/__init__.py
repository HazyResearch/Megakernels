from importlib import import_module
from pkgutil import iter_modules

for module_info in iter_modules(__path__):
    module = import_module(f".{module_info.name}", __name__)
    if module_info.ispkg:  # is this a folder with __init__.py?
        for sub_module_info in iter_modules(module.__path__):
            import_module(f".{sub_module_info.name}", module.__name__)
