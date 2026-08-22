from language_model.preflight import cuda_capability_supported


def test_cuda_capability_validation() -> None:
    assert cuda_capability_supported((7, 5), ["sm_70", "sm_75", "sm_80"])
    assert not cuda_capability_supported((6, 0), ["sm_70", "sm_75", "sm_80"])
    assert cuda_capability_supported((6, 0), [])
