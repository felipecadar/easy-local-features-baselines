'''
A super generic match util.
'''


class GenericMatcher():
    def __init__(self, method_name, **kwargs):
        self.matcher = None
        self.matcher_kwargs = kwargs

    def __call__(self, features1, features2):
        return self.matcher(features1, features2, **self.matcher_kwargs)