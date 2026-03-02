import itertools

_BUTTONS = ["A", "B", "left", "right", "down"]
COMPLEX_MOVEMENT = [["NOOP"]]
for r in range(1, len(_BUTTONS) + 1):
    for combo in itertools.combinations(_BUTTONS, r):
        if "left" in combo and "right" in combo:
            continue
        COMPLEX_MOVEMENT.append(list(combo))
print(f"Defined {len(COMPLEX_MOVEMENT)} actions in COMPLEX_MOVEMENT action space.")

ACTION_MEANINGS = list(COMPLEX_MOVEMENT)
NUM_ACTIONS = len(ACTION_MEANINGS)

def get_action_meanings():
    return list(ACTION_MEANINGS)


def get_num_actions() -> int:
    return NUM_ACTIONS


def apply_action_space(env):
    """Wrap environment with COMPLEX_MOVEMENT action space."""
    from nes_py.wrappers import JoypadSpace

    return JoypadSpace(env, ACTION_MEANINGS)
