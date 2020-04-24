import os
import json
# import wget

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
from . import hate_speech_classifier


# global variables
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')
celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config.from_object("project.config.Config")
db = SQLAlchemy(app)
api = Api(app, version='1.0',
          title='UGC API services',
          description='REST APIs for processing user-generated content')
ns = api.namespace('comments_api', description='REST services API for news comments')


# input and output definitions

hate_speech_single_input = api.model('HateSpeechSingleInput', {
    'text': fields.String(required=True, description='input text for classification')
})
hate_speech_single_output = api.model('HateSpeechSingleOutput', {
    'label': fields.String(required=True, description='predicted class'),
    'confidence': fields.Float(required=True, description='prediction confidence')
})

hate_speech_list_input = api.model('HateSpeechListInput', {
    'texts': fields.List(fields.String, required=True, description='input list of texts for classification')
})
hate_speech_list_output = api.model('HateSpeechListOutput', {
    'labels': fields.List(fields.String, required=True, description='list of predicted classes'),
    'confidences': fields.List(fields.Float, required=True, description='list of prediction confidences')
})

@ns.route('/hate_speech/')
class HateSpeechClassifier(Resource):
    @ns.doc('predict hate speech from single text')
    @ns.expect(hate_speech_single_input, validate=True)
    @ns.marshal_with(hate_speech_single_output)
    def post(self):
        label, confidence = hate_speech_classifier.predict([api.payload['text']])
        return {'label': label[0],
                'confidence': confidence[0]}

#    @api.doc(responses={200: 'Success', 400: 'Input Error', 500: 'Internal Server Error'})
#    @api.expect(hate_model, validate=True)


@ns.route('/hate_speech_list/')
class HateSpeechListClassifier(Resource):
    @ns.doc('predict hate speech from list of texts')
    @ns.expect(hate_speech_list_input, validate=True)
    @ns.marshal_with(hate_speech_list_output)
    def post(self):
        labels, confidences = hate_speech_classifier.predict(api.payload['texts'])
        return {'labels': labels,
                'confidences': confidences}


@app.route("/health/")
#@app.doc('get information about the health of this API')
def health():
    return api_functions.health()

@app.route("/documentation/")
#@app.doc('get Swagger documentation about this API')
def documentation():
    return api_functions.documentation()

# ==========================================================================================================================================================================
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
