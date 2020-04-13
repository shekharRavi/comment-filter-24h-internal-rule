import os
import json
import wget

from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    redirect,
    url_for
)
from flask_sqlalchemy import SQLAlchemy
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_restx import Api, Resource, fields, abort, reqparse

from celery import Celery
import celery.states as states

from . import api_functions


# global variables
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')
celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config.from_object("project.config.Config")
db = SQLAlchemy(app)
api = Api(app, version='1.0',
          title='API services',
          description='dockerized flask+flask_restx+gunicorn+celery+redis+postgres+nginx skeleton for REST APIs')
ns = api.namespace('rest_api', description='REST services API')


# ================================================================================================================================
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from time import clock

from .hate_speech_classifier import predict_ml_hs

# configuration
ROOT_FOLDER = os.path.dirname(__file__)
DIRECTORIES = {
                'ml_hate_speech_path': os.path.join(ROOT_FOLDER, 'models/ml_hate_speech_classifier')
               }

LOG_FILENAME = os.path.join(ROOT_FOLDER, 'logs/log.txt')
USER_INPUT_CLASSIFICATION = os.path.join(ROOT_FOLDER, 'logs/user_input_classification.txt')

# namespaces
ns_multilingual_hate = api.namespace('ml_hate_speech', description='Multilingual hate speech classifiers.')

# argument models
hate_model = api.model('hate_speech_model',
                       {'tweet': fields.List(fields.String())}, description="Tweets for classification")



# Load a trained model that you have fine-tuned
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = 'bert-base-multilingual-uncased'
model_file = os.path.join(DIRECTORIES['ml_hate_speech_path'], 'pytorch_model_epoch_20_seqlen_256.bin')


print(os.path.isfile(model_file))
if not os.path.isfile(model_file):
    print('Downloading model ...')
    os.system('sh ./models/ml_hate_speech_classifier/model_download.sh')
print(model_file)
if torch.cuda.is_available():
    model_state_dict = torch.load(model_file)
else:
    model_state_dict = torch.load(model_file, map_location='cpu')
model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=2)
model.to(device)
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)



@ns_multilingual_hate.route('/ml_bert')
class PredictMLHateSpeech(Resource):
    """Accepts the data, invokes the model and returns the labels."""

    @api.doc(responses={200: 'Success', 400: 'Input Error', 500: 'Internal Server Error'})
    @api.expect(hate_model, validate=True)
    def post(self):
        """
        Multilingual model that classifies offensive tweets as offensive (OFF) or not (NOT).
        """
        t0 = clock()
        data = request.json['tweet']
        # print(data)
        # print(type(data))

        try:
            predictions, certainties = predict_ml_hs(data, tokenizer, model, device)
            response = []

            for prediction, certainty in zip(predictions, certainties):
                temp_dict = {'Label': prediction, 'Certainty': certainty}
                response.append(temp_dict)
            t = clock() - t0
            print("Exec. time: %f" % t)
            return response
        except Exception as e:
            error_message = "PredictMLHateSpeech: " + str(e)
            print(error_message)
            response = {'error': 'internal server error'}
            return response, 500
# ==========================================================================================================================================================================
# # input and output definitions
# tokenizer_input = api.model('TokenizerInput', {
#     'text': fields.String(required=True, description='input text')
# })
# tokenizer_output = api.model('TokenizerOutput', {
#     'tokens': fields.List(fields.String, description='tokens')
# })
#
# doc_tokenizer_input = api.model('DocTokenizerInput', {
#     'texts': fields.List(fields.String, required=True, description='list of texts')
# })
# doc_tokenizer_output = api.model('DocTokenizerOutput', {
#     'tokenized_texts': fields.List(fields.List(fields.String), description='tokens')
# })
#
# # async
# check_task_input = api.model('CheckTaskInput', {
#     'task_id': fields.String(required=True, description='task ID')
# })
# check_task_output = api.model('CheckTaskOutput', {
#     'state': fields.String(description='task state'),
# })
#
# get_task_result_input = api.model('GetTaskResultInput', {
#     'task_id': fields.String(required=True, description='task ID')
# })
# get_task_result_output = api.model('GetTaskResultOutput', {
#     'result': fields.String(description='result as JSON string')
# })
#
# async_translate_input = api.model('AsyncTranslateInput', {
#     'text': fields.String(required=True, description='text to translate'),
#     'target_lang': fields.String(required=True, description='target language')
# })
# async_translate_output = api.model('AsyncTranslateOutput', {
#     'task_id': fields.String(description='task ID')
# })
#
#
# @ns.route('/tokenize_text')
# class TextTokenizer(Resource):
#     @ns.doc('tokenizes input text')
#     @ns.expect(tokenizer_input, validate=True)
#     @ns.marshal_with(tokenizer_output)
#     def post(self):
#         return {'tokens': api_functions.tokenize_text(api.payload['text'])}
#
#
# @ns.route('/tokenize_docs')
# class DocsTokenizer(Resource):
#     @ns.doc('tokenizes a list of texts')
#     @ns.expect(doc_tokenizer_input, validate=True)
#     @ns.marshal_with(doc_tokenizer_output)
#     def post(self):
#         return {'tokenized_texts': api_functions.tokenize_documents(api.payload['texts'])}
#
#
# @ns.route('/check_task')
# class CheckTask(Resource):
#     @ns.doc('checks the status of the task')
#     @ns.expect(check_task_input, validate=True)
#     @ns.marshal_with(check_task_output)
#     def post(self):
#         res = celery.AsyncResult(api.payload['task_id'])
#         return {'state': res.state}
#
#
# @ns.route('/get_task_result')
# class GetTaskResult(Resource):
#     @ns.doc('gets the result of a completed task')
#     @ns.expect(get_task_result_input, validate=True)
#     @ns.marshal_with(get_task_result_output)
#     def post(self):
#         res = celery.AsyncResult(api.payload['task_id'])
#         if res.state != states.SUCCESS:
#             abort(404, 'Cannot get result!', task_state=res.state)
#         else:
#             return {'state': res.state, 'result': res.get()}
#
#
# @ns.route('/async_translate_text')
# class AsyncTranslator(Resource):
#     @ns.doc('translates input text asynchronously')
#     @ns.expect(async_translate_input, validate=True)
#     @ns.marshal_with(async_translate_output, code=201)
#     def post(self):
#         task = celery.send_task('tasks.translate', args=[api.payload['text'], api.payload['target_lang']], kwargs={})
#         return {'task_id': task.id}
#
#
# # serving static content
# @app.route("/static/<path:filename>")
# def staticfiles(filename):
#     return send_from_directory(app.config["STATIC_FOLDER"], filename)
#
# @app.route("/media/<path:filename>")
# def mediafiles(filename):
#     return send_from_directory(app.config["MEDIA_FOLDER"], filename)
