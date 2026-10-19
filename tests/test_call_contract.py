import inspect

from apairo import FramePreprocessor, SequencePreprocessor

import apairo_preprocess


def _preprocessors():
    for name in apairo_preprocess.__all__:
        obj = getattr(apairo_preprocess, name)
        if inspect.isclass(obj) and issubclass(obj, (FramePreprocessor, SequencePreprocessor)):
            yield obj


def test_every_preprocessor_implements_call():
    # A class that implements process() still runs, through apairo's deprecated
    # alias, but warns once at import: a user sees one DeprecationWarning per class.
    classes = list(_preprocessors())
    assert classes
    assert [cls.__name__ for cls in classes if "process" in vars(cls)] == []
