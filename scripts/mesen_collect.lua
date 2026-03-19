-- Mesen data collection script
--
-- Streams per-frame data (screenshot + RAM + WRAM + palette + input) over TCP
-- to a Python receiver that saves session .npz files.
--
-- Usage:
--   1. Start the Python receiver:
--      python scripts/collect_mesen.py --output data/smb3
--   2. Open Mesen, load your ROM
--   3. Load this script via Debug > Script Window > Open
--   4. Play the game — frames are streamed automatically
--   5. Stop the script or close Mesen to finalize the session
--
-- Protocol (per frame):
--   [4]    magic "MESF"
--   [4]    frame number  (uint32 LE)
--   [1]    controller 1 input bitmask
--   [2048] internal RAM  ($0000-$07FF via cpuDebug)
--   [8192] work RAM      ($6000-$7FFF via cpuDebug)
--   [32]   palette RAM
--   [4]    PNG byte count (uint32 LE)
--   [N]    PNG data       (from emu.takeScreenshot)

local socket = require("socket.core")

-- ======================== CONFIG ========================
local HOST           = "127.0.0.1"
local PORT           = 7275
local RAM_SIZE       = 2048   -- $0000-$07FF
local WRAM_START     = 0x6000
local WRAM_SIZE      = 8192   -- $6000-$7FFF
local PALETTE_SIZE   = 32
local RETRY_INTERVAL = 120    -- frames between reconnect attempts

-- ======================== STATE =========================
local client   = nil
local frameNum = 0
local retryCountdown = 0

-- ======================== HELPERS =======================

local function pack_u32(n)
  return string.char(
    n % 256,
    math.floor(n / 256) % 256,
    math.floor(n / 65536) % 256,
    math.floor(n / 16777216) % 256
  )
end

-- Bulk-read a contiguous memory region into a Lua string.
local function readBlock(startAddr, size, memType)
  local t = {}
  for i = 0, size - 1 do
    t[i + 1] = string.char(emu.read(startAddr + i, memType))
  end
  return table.concat(t)
end

-- Encode controller-1 state as a single bitmask byte.
-- Bit layout matches nes_py convention:
--   0=A  1=B  2=Select  3=Start  4=Up  5=Down  6=Left  7=Right
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

-- ======================== CONNECTION ====================

local function tryConnect()
  local c = socket.tcp()
  c:settimeout(0.05)
  local ok, err = c:connect(HOST, PORT)
  if ok or err == "already connected" then
    c:settimeout(0.2)  -- allow short blocking sends
    client = c
    frameNum = 0
    emu.log("[collect] Connected to " .. HOST .. ":" .. PORT)
    return true
  end
  c:close()
  return false
end

local function disconnect()
  if client then
    pcall(function() client:close() end)
    client = nil
  end
end

-- ======================== FRAME CALLBACK ================

local function onEndFrame()
  -- Reconnect logic
  if not client then
    retryCountdown = retryCountdown - 1
    if retryCountdown <= 0 then
      retryCountdown = RETRY_INTERVAL
      tryConnect()
    end
    return
  end

  -- Internal RAM
  local ram = readBlock(0x0000, RAM_SIZE, emu.memType.cpuDebug)

  -- Work RAM (may not exist — read via CPU debug address space)
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

  -- Assemble packet
  local packet = "MESF"
    .. pack_u32(frameNum)
    .. string.char(inputByte)
    .. ram
    .. wram
    .. pal
    .. pack_u32(#png)
    .. png

  local _, err = client:send(packet)
  if err then
    emu.log("[collect] Send error: " .. tostring(err))
    disconnect()
    return
  end

  frameNum = frameNum + 1
end

-- ======================== INIT ==========================

tryConnect()
emu.addEventCallback(onEndFrame, emu.eventType.endFrame)
emu.addEventCallback(function() disconnect() end, emu.eventType.scriptEnded)

emu.log("[collect] Mesen data collector loaded — server " .. HOST .. ":" .. PORT)
