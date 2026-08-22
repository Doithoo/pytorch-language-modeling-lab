from language_model.training.sampler import ResumableBatchSampler


def test_resumable_batch_sampler_reconstructs_remaining_order() -> None:
    full_sampler = ResumableBatchSampler(11, 3, seed=17, shuffle=True)
    full = list(full_sampler)

    resumed_sampler = ResumableBatchSampler(11, 3, seed=17, shuffle=True)
    resumed_sampler.set_start_micro_step(2)
    remaining = list(resumed_sampler)

    assert remaining == full[2:]


def test_resumable_batch_sampler_restarts_epoch_deterministically() -> None:
    first = ResumableBatchSampler(8, 4, seed=17, shuffle=True)
    first_epoch = list(first)
    second_epoch = list(first)

    restarted = ResumableBatchSampler(8, 4, seed=17, shuffle=True)
    restarted.set_start_micro_step(2)

    assert first_epoch != second_epoch
    assert list(restarted) == second_epoch
