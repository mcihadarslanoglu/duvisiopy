
import duvisiopy.model_functions
import types


class Model():

    def __init__(self,
                 model
                 ):
        self.model = model

    def __new__(self, cls):
        for attr_name in dir(duvisiopy.model_functions):

            if attr_name not in dir(cls) and not attr_name.startswith("_"):

                attr = getattr(duvisiopy.model_functions, attr_name)
                if not isinstance(attr, types.ModuleType):

                    attr = attr.__get__(duvisiopy.model_functions,
                                        duvisiopy.model_functions.__class__)
                    print(attr_name)

                    setattr(cls, attr_name, attr)
        return cls
