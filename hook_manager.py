class Hook_Manager:
    def __init__(self):
        self.hook_dict = {}

    def register_hook(self, key, function):
        self.hook_dict[key] = function

    def call_hook(self, key, *args, **kwargs):
        if key in self.hook_dict:
            return self.hook_dict[key](*args, **kwargs)
        else:
            raise KeyError(f"Hook key '{key}' not found.")
