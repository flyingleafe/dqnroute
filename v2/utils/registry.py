import inspect


class MetaParent(type):

    def __init__(cls, name, base, params, **kwargs):
        super().__init__(name, base, params)
        is_base_class = cls.mro()[1] is object
        if is_base_class:
            base_class = cls
        else:
            base_class_found = False
            for key in cls.mro():
                if isinstance(key, MetaParent) and key.mro()[1] is object:
                    assert base_class_found is False, 'multiple base classes(bug)'
                    base_class = key
                    base_class_found = True
            assert base_class_found is True, f'no base class for {name}'

        if is_base_class:
            cls._subclasses = {}

        @classmethod
        def __init_subclass__(scls, config_name=None):
            super().__init_subclass__()
            if config_name is not None:
                if config_name in base_class._subclasses:
                    raise ValueError("Class with name `{}` is already registered".format(config_name))
                base_class._subclasses[config_name] = scls

        cls.__init_subclass__ = __init_subclass__

        @classmethod
        def parent_create_from_config(cls, config, **kwargs):
            if 'type' in config:
                return cls._subclasses[config['type']].create_from_config(config, **kwargs)
            else:
                raise ValueError('There is no `type` provided for the `{}` class'.format(name))

        # Take kwargs for the last initialized baseclass
        init_kwargs = {}
        for bcls in cls.mro()[:-1]:  # Look into all base classes except object
            if '__init__' not in bcls.__dict__:
                continue
            init_kwargs = inspect.signature(bcls.__init__).parameters
            break

        @classmethod
        def child_create_from_config(cls, config):
            kwargs = {}
            for key, argspec in init_kwargs.items():
                if key == 'self':
                    continue
                value = config.get(key, argspec.default)
                if value is inspect.Parameter.empty:
                    msg = 'There is no value for `{}.__init__` required field `{}` in config `{}`'
                    raise ValueError(msg.format(cls, key, config))
                kwargs[key] = value
            return cls(**kwargs)

        if 'create_from_config' not in cls.__dict__:
            cls.create_from_config = parent_create_from_config if is_base_class else child_create_from_config
