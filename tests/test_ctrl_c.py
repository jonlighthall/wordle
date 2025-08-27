#!/usr/bin/env python3
"""
Test script to verify KeyboardInterrupt handling in Wordle mode
"""
import subprocess
import signal
import time
import os

def test_keyboard_interrupt():
    print("Testing KeyboardInterrupt behavior in Wordle mode...")

    # Start the wordle program
    proc = subprocess.Popen(
        ['./run_wordle.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd='/home/jlighthall/examp/wordle'
    )

    try:
        # Send inputs to select Wordle mode
        inputs = "1\n3\n"  # Interactive mode, then Wordle mode
        proc.stdin.write(inputs)
        proc.stdin.flush()

        # Give it a moment to start
        time.sleep(2)

        # Send KeyboardInterrupt (Ctrl+C)
        proc.send_signal(signal.SIGINT)

        # Wait for output
        stdout, stderr = proc.communicate(timeout=5)

        print("Program output:")
        print(stdout)

        # Check if "Would you like to play again?" appears
        if "Would you like to play again?" in stdout:
            print("❌ FAIL: 'Play again' prompt still appears after Ctrl+C")
            return False
        else:
            print("✅ PASS: 'Play again' prompt skipped after Ctrl+C")
            return True

    except subprocess.TimeoutExpired:
        proc.kill()
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_keyboard_interrupt()
    exit(0 if success else 1)
