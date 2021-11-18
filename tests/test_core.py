import sys
sys.path.insert(0, '..')
from semaxis import CoreUtil

def test_core_load_conceptnet_axes():
    assert len(CoreUtil.load_conceptnet_antonyms_axes()) == 732

def test_core_load_plutchik_wheel_of_emotions_axes():
    assert len(CoreUtil.load_plutchik_wheel_of_emotions_axes()) == 24    

def test_load_wordnet_antonyms_axes():
    assert len(CoreUtil.load_wordnet_antonyms_axes()) == 1828