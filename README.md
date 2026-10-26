# Prosthetic Hand Control Simulator

A learning simulation that guesses a hand gesture (rest, open or fist) from a fake muscle signal. It uses synthetic data only. It is **not** a medical device.

## How it works

1. Make a fake muscle signal for each gesture.
2. Cut the signal into short windows and measure the average size of each window.
3. Learn the typical window size for each gesture.
4. Guess a new window by picking the gesture with the closest typical size.
5. Score the guesses on new signals and print a results table.

![Window sizes for a long and a short window](export/window_comparison.png)

With a window of 200 readings the three gestures stand apart. With a window of 5 readings, `open` and `fist` overlap and the classifier makes mistakes.

## How to run

```bash
git clone https://github.com/playfu1-d0ubt/prosthetic-hand-simulator.git
cd prosthetic-hand-simulator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py        # train, test and print results
python plot.py        # save the comparison chart
python -m pytest      # run the tests
```

## Limitations

- All data is fake. It is random noise, not real muscle signal, so real signals would be messier.
- The 100% score is easy to get. The three gestures are far apart, so the task is too simple.
- It uses one feature only: the average size of a window. Two gestures with the same loudness would look the same.
- A short window is quick but noisy. A long window is steady but slow to react.
- It is a learning simulation, not a medical device.
- Tested on Linux Mint only.

## What I learned

In Song Quiz, all my code was in one file. Here I split it into `signals.py`, which makes the fake signal, `features.py`, which turns the signal into one size number per window, and `classifier.py`, which learns the typical size of each gesture, guesses gestures and scores the guesses. This helped because it broke the project down into small groups of functions.

The bug that caught me most was a missing `return`. I noticed because `len()` failed with a `TypeError`, since the function gave back `None`. I fixed it by adding `return` to the last line.