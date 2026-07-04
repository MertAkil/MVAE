from mvae_mnist.cli import build_parser


def test_cli_accepts_zero_and_none_digit_conditions() -> None:
    parser = build_parser()

    zero_args = parser.parse_args(["sample", "--condition-label", "0"])
    assert zero_args.condition_label == 0

    none_args = parser.parse_args(["sample", "--condition-label", "none"])
    assert none_args.condition_label is None
