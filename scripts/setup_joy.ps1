# USB NES Controller Setup for Windows WSL
#
# Prerequisites:
#   - Install usbipd-win on Windows: winget install usbipd
#   - Install USB/IP tools in WSL:   sudo apt install linux-tools-generic hwdata
#
# Usage (run from an elevated PowerShell prompt on Windows):
#
#   1. List USB devices to find your controller's bus ID:
#        usbipd list
#
#   2. Attach the controller to WSL (replace 1-9 with your actual bus ID):
#        usbipd attach --wsl --busid 1-9
#
#   3. (Optional) Verify it appeared inside WSL:
#        lsusb
#
#   4. If your user can't read the device, fix permissions inside WSL:
#        sudo chmod 666 /dev/input/event*
#
#   5. Run the game:
#        python playsmb3.py
#
# To detach later:
#   usbipd detach --busid 1-9

# --- Quick attach helper ---
# Set this to your controller's bus ID from `usbipd list`
$BusId = "1-9"

Write-Host "Attaching USB device $BusId to WSL..."
usbipd attach --wsl --busid $BusId
Write-Host "Done. The controller should now be available inside WSL."