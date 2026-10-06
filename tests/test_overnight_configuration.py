from pathlib import Path

from scripts.profile_overnight_training import ProfilePopulation, select_profile_examples
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Split, StackedMlpProjectorConfig
from speech_projector.overnight_configuration import sweep_runs


class CountingWrapper(FrozenQwen):
    def target_token_count(self, example: Example) -> int:
        return len(example.target_text.split())


def test_controlled_sweep_has_six_rates_and_identical_effective_budget() -> None:
    configurations = sweep_runs(38194, 2)
    assert len(configurations) == 6
    assert len({config.name for config in configurations}) == 6
    assert [
        config.projector.native_rate / config.projector.compression_factor
        for config in configurations
    ] == [10, 5, 5, 10, 2.5, 25]
    assert (
        sum(isinstance(config.projector, StackedMlpProjectorConfig) for config in configurations)
        == 2
    )
    assert all(
        config.microbatch_size * config.gradient_accumulation == 8 for config in configurations
    )
    assert all(
        config.max_optimizer_updates == 2000 and config.max_target_tokens == 4097
        for config in configurations
    )
    assert (
        len({(config.seed, config.learning_rate, config.epochs) for config in configurations}) == 1
    )


def test_profile_worst_population_includes_target_and_audio_tails(tmp_path: Path) -> None:
    wrapper = CountingWrapper.__new__(CountingWrapper)
    wrapper.config = sweep_runs(20, 1)[0]
    rows: list[Example] = []
    for index in range(20):
        feature = tmp_path / f"{index}.pt"
        feature.touch()
        rows.append(
            Example(
                example_id=str(index),
                dialogue_id=str(index),
                split=Split.TRAIN,
                history=(),
                user_text="True words",
                target_text="word " * (20 - index),
                audio_path=tmp_path / f"{index}.wav",
                duration=index + 1,
                domain="test",
                emotion="",
                feature_path=feature,
            )
        )
    selected = select_profile_examples(wrapper, rows, ProfilePopulation.LONGEST)
    assert [row.example_id for row in selected] == ["0", "1", "2", "3", "19", "18", "17", "16"]
    first = select_profile_examples(wrapper, rows, ProfilePopulation.REPRESENTATIVE)
    assert first == select_profile_examples(wrapper, rows, ProfilePopulation.REPRESENTATIVE)
    assert len({row.example_id for row in first}) == 8
