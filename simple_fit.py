#!/usr/bin/env python3

import numpy as np		# Smaller utilities for data processing and numerics
import sys			# Needed for epsilon
import PySimpleGUI as sg	# Python GUI
import matplotlib		# Plotting library
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D	# needed for 'markers'
from scipy.optimize import curve_fit	# Fitting library


#### Define fit functions ####

def lin(x, a, b, c):
	return a*x + b

def exp(x, a, b, c):
	return a * np.exp(b * x) + c

def poly2(x, a, b, c):
	return a*x**2 + b*x + c

funcs = [exp, lin, poly2]
function_labels = ["Exp: A*exp(B*x)+C", "Linear: A*x+B", "Polynomial2: A*x**2 + B*x + C"]

# Define fit routine
def perform_fit(X, Y, guesses, x0, x1, lower, upper, func_id):

	# Get specific function
	func = funcs[func_id] 

	# Restrict X and Y data to the specified range
	Xr = [X[i] for i in range(len(X)) if x0 <= X[i] <= x1]
	Yr = [Y[i] for i in range(len(X)) if x0 <= X[i] <= x1]

	# Create a continuous range of X values for the fit curve
	N = len(Xr)
	Xrange = np.linspace(x0, x1, 100*N)

	# Compute the fit using the scipy.optimize routine:
	popt, pcov = curve_fit(func, Xr, Yr, p0=guesses, bounds=(lower, upper)) 
	fit_result = [Xrange, func(Xrange, *popt)]

	a,b,c = popt
	chi_sq_red = np.mean([pow(Yr[i]-func(Xr[i],a,b,c), 2)/pow(Yr[i], 2) for i in range(N)])
	return popt, pcov, chi_sq_red, fit_result

# Update GUI elements with initial guesses
def update_guess(win, current_param_guesses):
	elements = ['p1', 'p2', 'p3']
	for i in range(len(elements)):
		window.FindElement(elements[i]).Update(value=current_param_guesses[i])

# Update GUI elements with final results
def update_results(win, popt, pcov, chi_sq_red):
	popt = list(popt)
	errs = list(np.sqrt(np.diag(pcov)))
	elements = ['p1f', 'p2f', 'p3f', 'p1err', 'p2err', 'p3err', 'chi']
	vals = popt + errs + [chi_sq_red]
	for i in range(len(elements)):
		window.FindElement(elements[i]).Update(value=vals[i])

#### Create GUI ####

# Define settings for GUI and figure
sg.theme('Default 1')
fig = plt.gcf() 
ax = fig.add_subplot(111)
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

# GUI: Menu
menu_def = [['File', ['Reset','Exit']]]

# Draw data settings frame
s = (5,1)
upper_frame = sg.Frame(title='Input dataset', layout=
	[
		[sg.Text('File: ', s), 
		sg.Input(key='fn', size=(26,1), enable_events=1, disabled=1),
		sg.FileBrowse(font='Any 10')
		],

		[sg.Text('X col:', s), 
		sg.Combo(['None'], key='xcol', enable_events=1, readonly=1, size=(10,1)),
		sg.Checkbox('Log x', key='xlog', enable_events=1), 
		],

		[sg.Text('Y col:', s), 
		sg.Combo(['None'], key='ycol', enable_events=1, readonly=1, size=(10,1)),
		sg.Checkbox('Log y', key='ylog', enable_events=1), 
		sg.Text('| Symbol:'),
		sg.Combo([], key='marker', size=(4,1), readonly=1, enable_events=1)
		],

		#[sg.Text('Y err', s), sg.Combo(['None'], key='yerr', enable_events=1)]
	])

# Draw fit options frame
result_box = sg.Multiline('A = None +/- None\nB = None +/- None\nC = None +/- None', key='results1', disabled=1, size=(40, 3))
s3 = (3,1)
s4 = (23,1)
s5 = (16,1)
option_frame = sg.Frame(title='Fit options', size=(figure_w, figure_h), layout=
	[
		[sg.Text('Select fit function: ', (26,1))], 
		[sg.Listbox(function_labels, key='fit_func', size=(38, 3))], 
		
		[sg.Text('Initial parameter values:')],

		[sg.Text('A:',s3), 
		sg.Input(size=s4, key='p1', enable_events=1), 
		sg.Checkbox('Keep fixed', key='p1_const')
		],

		[sg.Text('B:',s3), 
		sg.Input(size=s4, key='p2', enable_events=1), 
		sg.Checkbox('Keep fixed', key='p2_const')
		],

		[sg.Text('C:',s3), 
		sg.Input(size=s4, key='p3', enable_events=1), 
		sg.Checkbox('Keep fixed', key='p3_const')
		],

		[sg.Text('Fit only from x ='), 
		sg.Input(key='x0', size=(5,1)), 
		sg.Text('to'), 
		sg.Input(key='x1', size=(5,1))
		],
		
		[sg.Checkbox('Show live preview', key='show_prev', enable_events=1), 
		sg.Checkbox('Auto update initial vals', key='auto_update_guess')
		], 

		[sg.OK('Perform the fit', key='perform_fit'), 
		sg.Button('Reset',key='Reset')
		]
	])

# Draw fit results frame
result_frame = sg.Frame(title='Fit results', size=(figure_w, figure_h), layout=
	[
		[sg.Text('=> Resulting parameter values:')],
		
		[sg.Text('A:',s3), 
		sg.Input(size=s5, key='p1f', disabled=1), 
		sg.Text('+/-'), 
		sg.Input(size=s5, disabled=1, key='p1err')
		],

		[sg.Text('B:',s3), 
		sg.Input(size=s5, key='p2f', disabled=1), 
		sg.Text('+/-'), 
		sg.Input(size=s5, disabled=1, key='p2err')
		],

		[sg.Text('C:',s3), 
		sg.Input(size=s5, key='p3f', disabled=1), 
		sg.Text('+/-'), 
		sg.Input(size=s5, disabled=1, key='p3err')
		],

		[sg.Text('χ²/d:',s3), 
		sg.Input(size=s5, key='chi', disabled=1)
		],

	])

# Create entire layout
layout = [
	[sg.Menu(menu_def)],
	[
		sg.Column([
			[upper_frame],
			[option_frame],
			[result_frame],
			[sg.Text('',size=(5,1)), sg.Button('Quit program')]
		]),
		sg.Column([
			[sg.Canvas(size=(figure_w, figure_h), key='fig_canvas')], 
			[sg.Canvas(size=(figure_w, 10), key='controls') ]
		])
	]]

# Create the GUI window
window = sg.Window('Simple fit', layout, force_toplevel=1, finalize=1, font="Any 12")

# Initialize GUI window elements with default values
window.FindElement('fit_func').Update(set_to_index=0)
window.FindElement('xcol').Update(set_to_index=0)
window.FindElement('ycol').Update(set_to_index=0)
window.FindElement('marker').Update(values=['o']+[str(x) for x in Line2D.markers if x != 'o'], value='o')
window.FindElement('p1').Update(value=1.)
window.FindElement('p2').Update(value=1.)
window.FindElement('p3').Update(value=1.)
window.FindElement('auto_update_guess').Update(value=True)

# Draw initial figure canvas and toolbar
canvas = window['fig_canvas'].TKCanvas
toolbar_elem = window['controls'].TKCanvas
figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
figure_canvas_agg.draw()
toolbar = NavigationToolbar2Tk(figure_canvas_agg, toolbar_elem)
toolbar.update()
figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

# Status variables and settings
class ProgramState:
	filename = None
	file_loaded = False
	cols = []
	column_labels = []
	input_data = []
	new_param_guesses = [1.,1.,1.]
	xcol = 0
	ycol = 0
	X = []
	Y = []
	fit_result = []
	popt = []
	pcov = []

pstate = ProgramState()

# Initialize plot
ax.cla()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.plot(pstate.X, pstate.Y, marker='o', linestyle='', color='tab:blue')
figure_canvas_agg.draw()


#### Event loop: Handle events and call functions: ###
while 1:

	event, values = window.read()
	ax.cla()

	# If window closes:
	if event in [None, 'Exit', 'Cancel', 'Quit program']:
		break

	# If user pressed 'perform fit'
	elif event == 'perform_fit' and len(pstate.X) != 0:

		# Erase previous plot
		ax.cla()

		# Obtain x range for fit (try user input first, otherwise adapt to data)
		try:
			x0 = float(values['x0'])
		except:
			x0 = min(pstate.X)
		try:
			x1 = float(values['x1'])
		except:
			x1 = max(pstate.X)

		# Obtain fit
		try:
			p1 = float(values['p1'])
			p2 = float(values['p2'])
			p3 = float(values['p3'])
			params = [p1, p2, p3]

			# Set lower and upper parameter bounds to (-)inf unless params are kept const...
			lower = [-np.inf for x in params]
			upper = [np.inf for x in params]

			# ...then we use the "bounds" feature to avoid variation of the respective param
			if values['p1_const']:
				lower[0] = p1-sys.float_info.epsilon
				upper[0] = p1+sys.float_info.epsilon
			if values['p2_const']:
				lower[1] = p2-sys.float_info.epsilon
				upper[1] = p2+sys.float_info.epsilon
			if values['p3_const']:
				lower[2] = p3-sys.float_info.epsilon
				upper[2] = p3+sys.float_info.epsilon
			
			# Get the specific fit function and perform the fit
			f = values['fit_func'][0]
			func_idx = function_labels.index(f)
			pstate.popt, pstate.pcov, chi_sq_red, pstate.fit_result = perform_fit(pstate.X,pstate.Y,params,x0,x1,lower,upper, func_idx)

			# Update the GUI output with the fit results
			update_results(window, pstate.popt, pstate.pcov, chi_sq_red)
			pstate.new_param_guesses = list(pstate.popt)
			if values['auto_update_guess']:
				update_guess(window, pstate.new_param_guesses)
		except:
			print("Error: fit didn't succeed!")

	# If user selected file
	elif event == 'fn':
		fn_new = values['fn']
		if fn_new != pstate.filename:
			if fn_new == '':
				continue
			try:
				pstate.input_data = np.genfromtxt(fn_new, float, names=True)
				pstate.column_labels = pstate.input_data.dtype.names
			except:
				print("Error: couldn't load file contents.")
				continue
			try:
				# Get file data and columns
				pstate.input_data = [list(x) for x in pstate.input_data]
				pstate.cols = list(range(len(pstate.input_data[0])))
				column_select_items = ["{}: '{}'".format(x, pstate.column_labels[x]) for x in range(len(pstate.cols))]

				# Update GUI elements with the column data:
				# ...Fill x combobox:
				for x in ['xcol', 'ycol']:
					window.FindElement(x).Update(values=column_select_items)
				window.FindElement('xcol').Update(set_to_index=0)
				# ...overwrite values entry:
				values['xcol'] = column_select_items[0]
				# ...Fill y combobox:
				yidx = 1 if len(pstate.cols) > 1 else 0
				window.FindElement('ycol').Update(set_to_index=yidx)
				values['ycol'] = column_select_items[yidx]
				
				# Enable x/y log checkboxes
				window.FindElement('xlog').Update(disabled=0)
				window.FindElement('ylog').Update(disabled=0)

				# Update filename
				pstate.filename = fn_new
				pstate.file_loaded = 1
			except:
				print("Error: couldn't process file contents.")
				continue

	# Remove fit curve and reset initial data
	elif event == 'Reset':
		pstate.fit_result = []
		for p in ['p1', 'p2', 'p3']:
			window.FindElement(p).Update(value='1.0')
		ax.cla()

	# If live preview is active and there was a change in the textfield or the checkbox, create it:
	elif pstate.file_loaded and event == 'show_prev' or (values['show_prev'] and event in ['p1', 'p2', 'p3']):

		# Compute preview (drawn below)
		if values['show_prev']:
			Xrange = np.linspace(min(pstate.X), max(pstate.X), 1000)
			func = funcs[function_labels.index(values['fit_func'][0])]
			try:
				pstate.fit_result = [Xrange, func(Xrange, float(values['p1']), float(values['p2']), float(values['p3']))]
			except:
				1
		else:
			pstate.fit_result = []

	# If file or columns changed, update the values for xcol, ycol, X, Y:
	if event in ['fn', 'xcol', 'ycol']:
		if pstate.file_loaded:
			try:
				pstate.xcol = int(values['xcol'].split(":")[0])
			except:
				pstate.xcol = 0
			try:
				pstate.ycol = int(values['ycol'].split(":")[0])
			except:
				pstate.ycol = 0
			pstate.X = [x[pstate.xcol] for x in pstate.input_data]
			pstate.Y = [x[pstate.ycol] for x in pstate.input_data]

	# Set linear or log scale depending on user input
	ax.set_xscale('log' if values['xlog'] else 'linear')
	ax.set_yscale('log' if values['ylog'] else 'linear')

	# Plot the file data
	ax.plot(pstate.X, pstate.Y, marker=values['marker'], linestyle='', color='tab:blue')

	# If a preview or fit curve exists, draw also this curve (in red)
	if len(pstate.fit_result) > 0:
		ax.plot(pstate.fit_result[0], pstate.fit_result[1], 'r-', linewidth=1) 

	# Try to set x and y labels according to columns:
	try:
		ax.set_xlabel(pstate.column_labels[int(values['xcol'].split(":")[0])])
	except:
		ax.set_xlabel("x")
	try:
		ax.set_ylabel(pstate.column_labels[int(values['ycol'].split(":")[0])])
	except:
		ax.set_ylabel("y")

	# Draw everything
	figure_canvas_agg.draw()

window.close(); del window
