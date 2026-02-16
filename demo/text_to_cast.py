#!/usr/bin/env python3
"""Convert colored text output to asciinema cast with simulated timing."""
import json, sys, time

lines = open(sys.argv[1]).readlines()
width = 75
height = 45

# Header
print(json.dumps({"version": 2, "width": width, "height": height, "timestamp": int(time.time()), "env": {"TERM": "xterm-256color"}}))

t = 0.0
for line in lines:
    # Add realistic timing
    stripped = line.rstrip('\n')
    if '───' in stripped or stripped.strip() == '':
        delay = 0.3
    elif '👤' in stripped or '🔵' in stripped or '🟣' in stripped:
        delay = 0.8
    elif '✅' in stripped or '❌' in stripped:
        delay = 0.6
    elif 'Calling' in stripped:
        delay = 0.4
    elif '$ synapse' in stripped:
        delay = 0.5
    elif '💾' in stripped:
        delay = 0.3
    else:
        delay = 0.2
    
    t += delay
    print(json.dumps([round(t, 3), "o", line]))
