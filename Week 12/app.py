from flask import Flask, request, render_template

from clean_data import * 

app = Flask(__name__)


## load the model
# Load the weights and biases from a file
weights_and_biases = torch.load('pytorch_model.bin')

# Create a new ClassificationModel object using the loaded weights and biases
roberta = ClassificationModel('roberta', 'roberta-base', state_dict=weights_and_biases, use_cuda=False)

@app.route('/')
def home():
	return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
	
	input = [clean_text(x) for x in request.form.values()]
	prediction = roberta.predict(input)
	
	if prediction[0].item() == 0:
		final_pred = 'Free Speech'
	else:
		final_pred = 'Hate Speech'

	return render_template('Index.html', prediction_text = 'The tweet is classified as {}'.format(final_pred))

if __name__ == '__main__':
	app.run(debug=True)