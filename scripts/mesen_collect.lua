-- Mesen data collection script
--
-- Writes per-frame data (screenshot + RAM + WRAM + palette + input) to a
-- binary file on disk.  Run the Python converter afterwards:
--
--   python scripts/collect_mesen.py --input <recording.bin> --output data/smb3
--
-- Usage:
--   1. Open Mesen, load your ROM
--   2. Load this script via Debug > Script Window > Open
--   3. Play the game — every frame is written to disk
--   4. Stop the script or close Mesen to finalize
--   5. Convert offline:  python scripts/collect_mesen.py ...
--
-- File format:
--   Header (5 bytes):
--     [4]  magic "MESD"
--     [1]  version = 1
--   Per frame:
--     [4]    frame number  (uint32 LE)
--     [1]    controller 1 input bitmask
--     [2048] internal RAM  ($0000-$07FF)
--     [8192] work RAM      ($6000-$7FFF)
--     [32]   palette RAM
--     [4]    PNG byte count (uint32 LE)
--     [N]    PNG data       (from emu.takeScreenshot)

-- ======================== CONFIG ========================
-- Set OUTPUT_DIR to control where recordings are saved.
-- Default: Mesen's per-script data folder.
local OUTPUT_DIR = nil   -- nil = use emu.getScriptDataFolder()

local RAM_SIZE       = 2048   -- $0000-$07FF
local WRAM_START     = 0x6000
local WRAM_SIZE      = 8192   -- $6000-$7FFF
local PALETTE_SIZE   = 32

-- ======================== STATE =========================
local outFile  = nil
local frameNum = 0

-- ======================== HELPERS =======================

local function pack_u32(n)
  return string.char(
    n % 256,
    math.floor(n / 256) % 256,
    math.floor(n / 65536) % 256,
    math.floor(n / 16777216) % 256
  )
end

local function readBlock(startAddr, size, memType)
  local t = {}
  for i = 0, size - 1 do
    t[i + 1] = string.char(emu.read(startAddr + i, memType))
  end
  return table.concat(t)
end

local function encodeInput()
  local inp = emu.getInput(0)
  local b = 0
  if inp.a      then b = b + 1   end
  if inp.b      then b = b + 2   end
  if inp.select then b = b + 4   end
  if inp.start  then b = b + 8   end
  if inp.up     then b = b + 16  end
  if inp.down   then b = b + 32  end
  if inp.left   then b = b + 64  end
  if inp.right  then b = b + 128 end
  return b
end

-- ======================== FILE I/O ======================

local function openRecording()
  local dir = OUTPUT_DIR or emu.getScriptDataFolder()

  -- Find next available filename
  local index = 0
  while true do
    local path = dir .. "/recording_" .. string.format("%04d", index) .. ".bin"
    local f = io.open(path, "rb")
    if f then
      f:close()
      index = index + 1
    else
      break
    end
  end

  local path = dir .. "/recording_" .. string.format("%04d", index) .. ".bin"
  outFile = io.open(path, "wb")
  if not outFile then
    emu.log("[collect] ERROR: cannot open " .. path)
    return
  end

  -- Write header
  outFile:write("MESD")   -- magic
  outFile:write("\1")     -- version 1
  outFile:flush()

  frameNum = 0
  emu.log("[collect] Recording to: " .. path)
end

local function closeRecording()
  if outFile then
    outFile:flush()
    outFile:close()
    emu.log("[collect] Closed recording — " .. frameNum .. " frames saved")
    outFile = nil
  end
end

-- ======================== FRAME CALLBACK ================

local function onEndFrame()
  if not outFile then return end

  -- Internal RAM
  local ram = readBlock(0x0000, RAM_SIZE, emu.memType.cpuDebug)

  -- Work RAM (may not exist for all mappers)
  local wram
  local ok, result = pcall(readBlock, WRAM_START, WRAM_SIZE, emu.memType.cpuDebug)
  if ok then
    wram = result
  else
    wram = string.rep("\0", WRAM_SIZE)
  end

  -- Palette
  local pal = readBlock(0, PALETTE_SIZE, emu.memType.palette)

  -- Input
  local inputByte = encodeInput()

  -- Screenshot (PNG binary)
  local png = emu.takeScreenshot()

  -- Write frame record
  outFile:write(pack_u32(frameNum))
  outFile:write(string.char(inputByte))
  outFile:write(ram)
  outFile:write(wram)
  outFile:write(pal)
  outFile:write(pack_u32(#png))
  outFile:write(png)

  frameNum = frameNum + 1

  if frameNum % 300 == 0 then
    outFile:flush()
    emu.log("[collect] " .. frameNum .. " frames written")
  end
end

-- ======================== INIT ==========================

openRecording()
emu.addEventCallback(onEndFrame, emu.eventType.endFrame)
emu.addEventCallback(closeRecording, emu.eventType.scriptEnded)

emu.log("[collect] Mesen data collector loaded — writing to disk")
