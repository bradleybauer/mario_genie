# Xbox / USB Controller Setup for Windows WSL2
#
# The stock WSL2 kernel lacks the xpad driver, so Xbox One controllers
# won't create /dev/input/event* devices.  The gamepad module works
# around this by reading the controller directly over USB via pyusb.
#
# Prerequisites:
#   Windows:
#     winget install usbipd
#
#   WSL (one-time):
#     sudo apt install linux-tools-generic hwdata
#     pip install pyusb            # in your conda env
#
# Usage (from an elevated PowerShell prompt):
#
#   1. Bind your controller (first time only):
#        usbipd bind --busid 1-9
#
#   2. Attach it to WSL (repeat after each replug / WSL restart):
#        usbipd attach --wsl --busid 1-9
#
#   3. Fix USB permissions inside WSL so you don't need sudo:
#        sudo chmod 666 /dev/bus/usb/$(lsusb | grep -i xbox | awk '{print $2"/"$4}' | tr -d ':')
#
#      Or use the udev rule (persists across replugs):
#        echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="24c6", ATTR{idProduct}=="581a", MODE="0666"' \
#          | sudo tee /etc/udev/rules.d/99-xbox-controller.rules
#        sudo udevadm control --reload-rules
#
#   4. Run the game:
#        python scripts/collect.py --mode human --world 1 --stage 1
#
#   You should see: "Initialized Xbox One controller via USB (24c6:581a)"
#
# Troubleshooting:
#   - "No gamepad found" → check lsusb shows the controller, then fix
#     /dev/bus/usb permissions (step 3).
#   - "Device is not shared" → run usbipd bind first (step 1).
#   - For non-Xbox USB gamepads (e.g. generic HID): the evdev path
#     should work.  Ensure /dev/input/event* is readable:
#       sudo chmod 666 /dev/input/event*
#
# To detach later:
#   usbipd detach --busid 1-9

# --- Quick attach helper ---
# Set this to your controller's bus ID from `usbipd list`
$BusId = "1-9"

Write-Host "Binding USB device $BusId (first-time only, may already be bound)..."
usbipd bind --busid $BusId 2>$null

Write-Host "Attaching USB device $BusId to WSL..."
usbipd attach --wsl --busid $BusId
Write-Host "Done. Run inside WSL to fix permissions:"
Write-Host "  sudo chmod 666 /dev/bus/usb/`$(lsusb | grep -i xbox | awk '{print `$2`"/`"`$4}' | tr -d ':')"