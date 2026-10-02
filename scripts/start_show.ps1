# Starts the FluxRT server on a Windows show machine. Run it from anywhere; Ctrl+C stops it.
#
#   scripts\start_show.ps1                      production look: flow upscaler, 1152x640 out
#   scripts\start_show.ps1 -Fast                flow upscaler + tighter change mask (fewer tokens recomputed)
#   scripts\start_show.ps1 -Fast -NoUpscaler    fastest: tighter change mask, no upscaler, 576x320 out
#   scripts\start_show.ps1 -Fast --set gpu_wait=sleep     anything else goes to run_webrtc.py as is
#
# --fast / fast and --no-upscaler / no-upscaler work too.
# Settings that hold for every start live in the config (default configs\show_laptop.json):
# int8_linear, rife_cudagraphs, conv_backend, gpu_wait, lip transfer off. See docs\perf-notes.md.
# Boot takes about 2 min (warm-up); the first boot after a code or model change takes minutes longer.
[CmdletBinding(PositionalBinding = $false)]
param(
  [switch]$Fast,
  [switch]$NoUpscaler,
  [string]$Config = "configs\show_laptop.json",
  # everything else, e.g. --set gpu_wait=sleep (must not land in -Config)
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$extra = @()
foreach ($a in $Rest) {
  if ($a -match '^-{0,2}fast$') { $Fast = $true }
  elseif ($a -match '^-{0,2}no-?upscaler$') { $NoUpscaler = $true }
  else { $extra += $a }
}
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\.venv\Scripts\activate.ps1
$flags = @("--no-server-camera", "--tiny-vae", "--interp", "1", "--config", $Config)
if ($Fast) { $flags += @("--set", "mask_dilation=1") }
if (-not $NoUpscaler) { $flags += "--flow-upscaler" }
"{0} preset: {1} | config {2}{3}" -f $(if ($Fast) { "FAST" } else { "PRODUCTION" }), $(if ($NoUpscaler) { "no upscaler, 576x320 out" } else { "flow upscaler, 1152x640 out" }), $Config, $(if ($extra) { " | " + ($extra -join ' ') } else { "" })
# FLUXRT_DRY_RUN=1: print the command instead of starting the server
if ($env:FLUXRT_DRY_RUN) { "python scripts\run_webrtc.py " + (($flags + $extra) -join ' '); return }
python scripts\run_webrtc.py @flags @extra
