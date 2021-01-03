from flask import Flask, render_template, request, redirect, url_for
import csv
app = Flask(__name__)

@app.route('/')

def index():
	return render_template('home.html')

@app.route('/your-url', methods=['GET','POST'])

def your_url():
	if request.method == "POST":
		mealtime = request.form['mealtime']
		bedtime = request.form['bedtime']
		waketime = request.form['waketime']
		electronics = request.form['electronics']
		up = request.form['up']
		temperature = request.form['temperature']
		noise = request.form['noise']
		nap = request.form['nap']
		quality = request.form['electronics']

		fieldnames =['mealtime','bedtime' , 'waketime' , 'quality', 'electronics' ,'up', 'temperature', 'noise', 'nap']
		with open('namelist.csv','a', newline='') as inFile:
			writer = csv.DictWriter(inFile, fieldnames=fieldnames)
			writer.writerow({'mealtime':mealtime,'bedtime':bedtime, 'waketime':waketime, 'quality':quality , 'electronics':electronics, 'up':up, 'temperature':temperature, 'noise':noise, 'nap':nap})

		#return 'Thanks for your input'\
		return render_template('your_url.html', mealtime =request.form['mealtime'])
	else:
		return redirect(url_for('home'))

if __name__ == "__main__":
	app.run()
