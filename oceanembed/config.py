"""YAML run configs (configs/*.yaml) -> argparse defaults. Command-line flags still override."""
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for key, value in cfg.items():                     # allow one level of grouping
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def apply_config(parser, argv):
    """If --config is given, use the file's values as defaults, then parse normally."""
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "config", None):
        cfg = load_config(pre.config)
        unknown = set(cfg) - {a.dest for a in parser._actions}
        if unknown:
            raise KeyError(f"Unknown keys in {pre.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)
