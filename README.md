# simplefit (beta)
Python interface for fitting exponential, linear and polynomial trends with scipy.optimize and pysimplegui, originally developed to fit pandemic growth. Currently still in development phase.

# Requirements
Python libraries: 
- numpy
- scipy
- matplotlib
- pysimplegui

# Usage
Run the code with no arguments. This should open a GUI window in which you can import text catalog files with multiple columns (requires header):

![GUI](http://sebastian.stapelberg.de/documents/simplefit.png "GUI")

The catalog should be a file structured like this:

``` shell
#x y1 y2 ...
1  10 10 ...
2  30 35 ...
.  .  .  . 
.  .  .  .
.  .  .  .
```

After importing the data, you should immediately see a plot. Select the x and y column you want to use and specify the fit options. If the fit doesn't converge, you can try different initial parameter guesses. There is also a live preview option, where you can see a plot of the function with the initial parameter values in realtime as you type them in.

# License

MIT License
