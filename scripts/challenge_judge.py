"""Independent predeclared response-quality challenges outside the rubric anchors."""

import argparse
from pathlib import Path

import torch

from scripts.run_judge import CalibrationCase, CalibrationCategory, CalibrationSummary, calibrate
from speech_projector.judge import JudgeConfig, JudgeRequest, LocalJudge
from speech_projector.models import Record, Role, Turn


class JudgeChallengeReport(Record):
    calibration: CalibrationSummary
    independent_challenge: CalibrationSummary

    @property
    def passed(self) -> bool:
        return self.calibration.passed and self.independent_challenge.passed


def challenge_cases() -> tuple[CalibrationCase, ...]:
    specific = (
        (
            (),
            "My friend Nora's birthday is Saturday. Write one short birthday greeting to her.",
            "Happy birthday, Nora! Hope you have a wonderful day.",
            "SQL indexes can make database queries faster.",
            "Celebrations are fun; I enjoy hearing about people's plans.",
            "Happy birthday, Ava! Hope you have a wonderful day.",
        ),
        (
            (
                Turn(role=Role.USER, text="I am moving to London soon."),
                Turn(role=Role.ASSISTANT, text="You mentioned moving to London."),
            ),
            "Actually, plans changed: I'm moving to Rome. Which city should I put on my form?",
            "Use Rome on the form, since your plan changed.",
            "I like the sound of rain on a quiet morning.",
            "Moving can be such an exciting adventure! Hope it goes smoothly.",
            "Put London on the form, as you said earlier.",
        ),
        (
            (
                Turn(role=Role.USER, text="I adopted a cat named Milo."),
                Turn(role=Role.ASSISTANT, text="Milo sounds lovely."),
            ),
            "Suggest one kind of toy for him.",
            "A wand toy could be fun for Milo; use it when playing together.",
            "The conference agenda has three keynote sessions.",
            "Pets can be so rewarding! I hope you enjoy your time together.",
            "Your new puppy Luna might enjoy a dog chew toy.",
        ),
        (
            (),
            "I don't want any eggs. Suggest a breakfast with no egg ingredients.",
            "Try oatmeal made with water and topped with fruit.",
            "An orchestra has several families of instruments.",
            "Breakfast is an important start to the day. Enjoy your morning!",
            "Make an omelet with two eggs and herbs.",
        ),
        (
            (),
            "Our workshop moved from Bergen to Oslo. Which city should invitees travel to?",
            "They should travel to Oslo, the workshop's new location.",
            "My favorite pastime is collecting colorful stamps.",
            "That workshop sounds exciting! I hope everyone enjoys it.",
            "Invitees should travel to Bergen for the workshop.",
        ),
    )
    cases: list[CalibrationCase] = []
    categories = (
        CalibrationCategory.PARAPHRASE,
        CalibrationCategory.UNRELATED,
        CalibrationCategory.GENERIC,
        CalibrationCategory.WRONG_ENTITY,
    )
    for index, (history, transcript, *responses) in enumerate(specific):
        for category, candidate in zip(categories, responses, strict=True):
            cases.append(
                CalibrationCase(
                    category=category,
                    request=JudgeRequest(
                        example_id=f"challenge-specific-{index}-{category.value}",
                        dialogue_id=f"challenge-specific-{index}",
                        history=history,
                        transcript=transcript,
                        candidate_response=candidate,
                    ),
                )
            )
    social = (
        (
            "I finally finished cleaning the kitchen!",
            "Nice work! Enjoy having that job finished.",
            "A router forwards packets between networks.",
        ),
        (
            "My weekend is quiet, and I like that.",
            "That sounds lovely. Hope you enjoy the peaceful weekend.",
            "Your refrigerator temperature is definitely broken.",
        ),
        (
            "I started painting for fun.",
            "That sounds enjoyable! Hope you have fun exploring it.",
            "The company's quarterly revenue increased.",
        ),
        (
            "I can't wait to meet my friends later.",
            "Have a great time with them!",
            "This paragraph describes volcanic rock formation.",
        ),
        (
            "My train trip yesterday was relaxing.",
            "Glad you had a relaxing trip!",
            "You should bring pepperoni to the costume party.",
        ),
    )
    for index, (transcript, friendly, unrelated) in enumerate(social):
        for category, candidate in (
            (CalibrationCategory.CONVERSATIONAL, friendly),
            (CalibrationCategory.UNRELATED, unrelated),
        ):
            cases.append(
                CalibrationCase(
                    category=category,
                    request=JudgeRequest(
                        example_id=f"challenge-social-{index}-{category.value}",
                        dialogue_id=f"challenge-social-{index}",
                        history=(),
                        transcript=transcript,
                        candidate_response=candidate,
                    ),
                )
            )
    return tuple(cases)


def run_challenge(judge: LocalJudge, output: Path) -> JudgeChallengeReport:
    output.mkdir(parents=True, exist_ok=True)
    report = JudgeChallengeReport(
        calibration=calibrate(judge, output / "calibration"),
        independent_challenge=calibrate(
            judge, output / "independent_challenge", cases=challenge_cases()
        ),
    )
    (output / "challenge_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output / "interpretation.md").write_text(
        "# Independent judge challenge\n\n"
        "The 30 challenge cases were declared before inspecting this checkpoint's judgments. "
        "They are absent from the judge's rubric examples and use distinct topics. Labels "
        "are model-assisted benchmark annotations, not a human certification. Context "
        "changes, named entities, ingredient constraints, direct requests, and valid brief "
        "social acknowledgments are represented. The unchanged rubric is used for both "
        "the 13 original calibration cases and independent challenge. Raw attempts/failures "
        "and per-category acceptance are retained. Passing both is a minimal quality gate, "
        "not proof of reliable grading on all 512 conversational samples.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    config = JudgeConfig(
        model_name=arguments.model, revision=arguments.revision, batch_size=arguments.batch_size
    )
    report = run_challenge(LocalJudge(config, torch.device("cuda")), arguments.output)
    print(report.model_dump_json(indent=2), flush=True)
    if not report.passed:
        raise ValueError("Judge failed original or independent gate; no full quality scores")


if __name__ == "__main__":
    main()
