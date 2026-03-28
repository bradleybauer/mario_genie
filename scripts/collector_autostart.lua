-- collector_autostart.lua
-- Launched by the targeted data collection driver.
-- The driver writes the output base path on the line below before each launch.
-- !! DO NOT EDIT the next line manually — it is machine-managed !!
local output_base = "__OUTPUT_BASE_PLACEHOLDER__"

if output_base == "" or output_base:find("PLACEHOLDER") then
  emu.log("ERROR: output_base was not set by the driver script")
  return
end

local started = false
emu.addEventCallback(function()
  if started then return end
  started = true
  -- Save state interval of 60 frames (~1 Hz at 60fps)
  emu.startDataRecording(output_base, 60)
  emu.log("Data recording started: " .. output_base)
end, emu.eventType.startFrame)
