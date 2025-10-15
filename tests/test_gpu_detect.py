from libs.gpu.detect import detect_all


def test_detect_all_imports_and_runs():
    det = detect_all()
    assert isinstance(det, dict)
