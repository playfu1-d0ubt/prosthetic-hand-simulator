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