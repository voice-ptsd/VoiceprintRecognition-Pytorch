class DictClass(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def __init__(self, dict_obj):
        super().__init__()
        self.__to_class(dict_obj)

    def __to_class(self, dict_obj):
        if not isinstance(dict_obj, dict):
            return dict_obj
        for k, v in dict_obj.items():
            self[k] = self.__to_class(v)
