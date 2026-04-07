from importlib import import_module
from pkgutil import iter_modules

for _info in iter_modules(__path__):
    import_module(f".{_info.name}", __name__)
