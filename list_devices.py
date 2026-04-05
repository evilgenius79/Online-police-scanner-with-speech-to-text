#!/usr/bin/env python3
"""
List all available audio INPUT devices.

Run this script first to find the correct AUDIO_DEVICE_INDEX value
for your Uniden scanner connection:

    python list_devices.py

Then open config.py and set:

    AUDIO_DEVICE_INDEX = <the index shown for your mic/line-in device>
"""
import sys

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.  Run:  pip install sounddevice")
    sys.exit(1)

print("\nAvailable audio INPUT devices:\n")
print(f"  {'IDX':>4}  {'SAMPLERATE':>11}  {'CH':>3}  NAME")
print("  " + "─" * 70)

for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] < 1:
        continue
    default = " <-- default" if i == sd.default.device[0] else ""
    print(
        f"  [{i:>3}]  "
        f"{int(dev['default_samplerate']):>10} Hz  "
        f"{dev['max_input_channels']:>3} ch  "
        f"{dev['name']}{default}"
    )

print()
print("Set AUDIO_DEVICE_INDEX in config.py to the [IDX] of your scanner input.")
print("Leave it as None to use the system default input device.")
print()
