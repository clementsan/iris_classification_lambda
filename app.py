import os
import requests
import gradio as gr
from classification.classifier import Classifier
from dotenv import load_dotenv, find_dotenv
import json

# Initialize API URLs from env file or global settings
def retrieve_api():

    env_path = find_dotenv('config_api.env')
    if env_path:
        load_dotenv(dotenv_path=env_path)
        print("config_api.env file loaded successfully.")
    else:
        print("config_api.env file not found.")

    # Use of AWS endpoint or local container by default
    global AWS_API
    AWS_API = os.getenv("AWS_API", default="http://localhost:8000")

def initialize_classifier():
    global cls
    cls = Classifier()


def predict_class_local(sepl, sepw, petl, petw):
    data = list(map(float, [sepl, sepw, petl, petw]))
    results = cls.load_and_test(data)
    return results


def predict_class_aws(sepl, sepw, petl, petw):
    if AWS_API == "http://localhost:8080":
        API_endpoint = AWS_API + "/2015-03-31/functions/function/invocations"
    else:
        API_endpoint = AWS_API + "/test/classify"

    data = list(map(float, [sepl, sepw, petl, petw]))
    json_object = {
        "features": [
            data
        ]
    }

    response = requests.post(API_endpoint, json=json_object)
    if response.status_code == 200:
        # Process the response
        response_json = response.json()
        results_dict = json.loads(response_json["body"])
    else:
        results_dict = {"Error": response.status_code}
        gr.Error(f"\t API Error: {response.status_code}")
    return results_dict


def predict(sepl, sepw, petl, petw, type):
    print("type: ", type)
    if type == "Local":
        results = predict_class_local(sepl, sepw, petl, petw)
    elif type == "AWS API":
        results = predict_class_aws(sepl, sepw, petl, petw)

    prediction = results["predictions"][0]
    confidence = max(results["probabilities"][0])

    return f"Prediction: {prediction} \t - \t Confidence: {confidence:.3f}"


# Define the Gradio interface
def user_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# IRIS classification task - use of AWS Lambda")
        gr.Markdown(
            """
            Aims: Categorization of different species of iris flowers (Setosa, Versicolor, and Virginica) 
            based on measurements of physical characteristics (sepals and petals).

            Notes: This web application uses two types of predictions:
               - local prediction (direct source code) 
               - cloud prediction via an AWS API (i.e. use of ECR, Lambda function and API Gateway) to run the machine learning model.
            
            """
        )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr_sepl = gr.Slider(minimum=4.0, maximum=8.0, step=0.1, label="Sepal Length (in cm)")
                    gr_sepw = gr.Slider(minimum=2.0, maximum=5.0, step=0.1, label="Sepal Width (in cm)")
                    gr_petl = gr.Slider(minimum=1.0, maximum=7.0, step=0.1, label="Petal Length (in cm)")
                    gr_petw = gr.Slider(minimum=0.1, maximum=2.8, step=0.1, label="Petal Width (in cm)")
            with gr.Column():
                with gr.Row():
                    gr_type = gr.Radio(["Local", "AWS API"], value="Local", label="Prediction type")
                with gr.Row():
                    gr_output = gr.Textbox(label="Prediction output")

        with gr.Row():
            submit_btn = gr.Button("Submit")
            clear_button = gr.ClearButton()

        submit_btn.click(fn=predict, inputs=[gr_sepl, gr_sepw, gr_petl, gr_petw, gr_type], outputs=[gr_output])
        clear_button.click(lambda: None, inputs=None, outputs=[gr_output], queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    retrieve_api()
    initialize_classifier()
    user_interface()
