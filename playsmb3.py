
import stable_retro
import pygame
import numpy as np

# --- Configuration ---
GAME_NAME = 'SuperMarioBros3-Nes-v0'
import select

try:
    import evdev
except ImportError:
    evdev = None
    print("Warning: 'evdev' not installed. Controller support disabled. Run 'pip install evdev'.")

# --- Configuration ---
SCALE_FACTOR = 3  
FPS = 60          

class RetroGamepadController:
    """Reads keyboard and raw evdev Gamepad inputs and converts them to stable-retro actions."""
    def __init__(self, env_buttons):
        self.env_buttons = env_buttons
        
        self._evdev_device = None
        self._evdev_axes = {"ABS_X": 128, "ABS_Y": 128}
        self._evdev_buttons = set()
        self._evdev_axis_ranges = {}
        
        self._init_evdev_gamepad()

    def _init_evdev_gamepad(self) -> None:
        if evdev is None: return
        try:
            paths = evdev.list_devices()
            if not paths:
                print("No evdev input devices found. Falling back to keyboard.")
                return
            for path in paths:
                try:
                    dev = evdev.InputDevice(path)
                except PermissionError:
                    print(f"Permission denied: {path} (Try: sudo chmod 666 {path})")
                    continue
                
                caps = dev.capabilities(verbose=False)
                has_abs = evdev.ecodes.EV_ABS in caps
                has_key = evdev.ecodes.EV_KEY in caps
                
                if has_abs or has_key:
                    self._evdev_device = dev
                    for abs_code, abs_info in caps.get(evdev.ecodes.EV_ABS,[]):
                        name = evdev.ecodes.ABS.get(abs_code, f"ABS_{abs_code}")
                        if isinstance(name, list): name = name[0]
                        mid = (abs_info.min + abs_info.max) // 2
                        self._evdev_axes[name] = mid
                        self._evdev_axis_ranges[name] = (abs_info.min, abs_info.max)
                    
                    print(f"Initialized Gamepad: {dev.name} ({dev.path})")
                    return
        except Exception as exc:
            print(f"evdev gamepad probe failed: {exc}")

    def _poll_evdev(self) -> None:
        """Non-blocking read of pending evdev events."""
        dev = self._evdev_device
        if dev is None: return
        try:
            while select.select([dev.fd], [],[], 0)[0]:
                for event in dev.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = evdev.ecodes.ABS.get(event.code, f"ABS_{event.code}")
                        if isinstance(name, list): name = name[0]
                        self._evdev_axes[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        if event.value >= 1:
                            self._evdev_buttons.add(event.code)
                        else:
                            self._evdev_buttons.discard(event.code)
        except (OSError, IOError):
            print("Gamepad disconnected!")
            self._evdev_device = None

    def get_action(self, keys) -> np.ndarray:
        # 1. Update Gamepad State
        self._poll_evdev()
        
        # 2. Base Keyboard states
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        jump = keys[pygame.K_x] or keys[pygame.K_o]
        sprint = keys[pygame.K_z] or keys[pygame.K_p]
        start = keys[pygame.K_RETURN]
        select_btn = keys[pygame.K_RSHIFT]

        # 3. Override with Gamepad states (if connected)
        if self._evdev_device is not None:
            def _axis_norm(axis_name: str) -> float:
                ax_min, ax_max = self._evdev_axis_ranges.get(axis_name, (0, 255))
                ax_mid = (ax_min + ax_max) / 2.0
                ax_half = max((ax_max - ax_min) / 2.0, 1)
                return (self._evdev_axes.get(axis_name, ax_mid) - ax_mid) / ax_half

            # Analog sticks
            x_val = _axis_norm("ABS_X")
            y_val = _axis_norm("ABS_Y")
            right = right or x_val > 0.5
            left = left or x_val < -0.5
            up = up or y_val < -0.5
            down = down or y_val > 0.5

            # D-Pad (HAT)
            hat_x = _axis_norm("ABS_HAT0X")
            hat_y = _axis_norm("ABS_HAT0Y")
            right = right or hat_x > 0.5
            left = left or hat_x < -0.5
            up = up or hat_y < -0.5
            down = down or hat_y > 0.5

            # Buttons (Checking both generic budget USB IDs and standard Xbox/PS IDs)
            btns = self._evdev_buttons
            jump = jump or any(b in btns for b in[
                getattr(evdev.ecodes, 'BTN_SOUTH', 304), getattr(evdev.ecodes, 'BTN_THUMB', 289)
            ])
            sprint = sprint or any(b in btns for b in[
                getattr(evdev.ecodes, 'BTN_WEST', 308), getattr(evdev.ecodes, 'BTN_EAST', 305), 
                getattr(evdev.ecodes, 'BTN_TRIGGER', 288)
            ])
            start = start or any(b in btns for b in[getattr(evdev.ecodes, 'BTN_START', 315)])
            select_btn = select_btn or any(b in btns for b in[getattr(evdev.ecodes, 'BTN_SELECT', 314)])

        # Prevent pressing opposing directions simultaneously
        if right and left: left = right = False
        if up and down: up = down = False

        # 4. Map to stable-retro's strict boolean action array
        logical_state = {
            'UP': up, 'DOWN': down, 'LEFT': left, 'RIGHT': right,
            'A': jump, 'B': sprint, 'START': start, 'SELECT': select_btn
        }

        action_array = np.zeros(len(self.env_buttons), dtype=np.int8)
        for i, button_name in enumerate(self.env_buttons):
            if logical_state.get(button_name, False):
                action_array[i] = 1

        return action_array


def main():
    print(f"Loading {GAME_NAME}...")
    try:
        env = stable_retro.make(game=GAME_NAME)
    except FileNotFoundError:
        print(f"Error: ROM not found.")
        return

    obs, info = env.reset()
    height, width, _ = obs.shape
    
    pygame.init()
    screen = pygame.display.set_mode((width * SCALE_FACTOR, height * SCALE_FACTOR))
    pygame.display.set_caption("Super Mario Bros 3 - Data Recorder")
    clock = pygame.time.Clock()

    # Initialize our new Controller Handler
    controller = RetroGamepadController(env.buttons)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Get perfectly formatted stable-retro action array from our controller logic
        action = controller.get_action(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        print(info) # Streaming metadata for your World Model

        if terminated or truncated:
            obs, info = env.reset()

        surface_array = np.swapaxes(obs, 0, 1)
        surface = pygame.surfarray.make_surface(surface_array)
        scaled_surface = pygame.transform.scale(surface, (width * SCALE_FACTOR, height * SCALE_FACTOR))
        
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
