import json
import boto3
import io
from PIL import Image
import ast


def img_to_byte(img):
    img = Image.open(img, mode='r')

    img_to_byte = io.BytesIO()

    img.save(img_to_byte, format='PNG')
    img_to_byte = img_to_byte.getvalue()

    return img_to_byte


def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = 'sagemaker-pytorch-2020-03-14-15-27-31-449',    # The name of the endpoint we created
                                       ContentType = 'application/x-image',                 # The data format that is expected
                                       Accept = 'application/json',
                                       Body = img_to_byte(event))                       # The image transformed to byte

    # The response is an HTTP response whose body contains the result of our inference
    rbody = response['Body']

    #transforming the output to the desired format (dict first, and then the right key)
    rbodystr = rbody.read().decode("UTF-8") #to str
    rbodydict = ast.literal_eval(rbodystr) #to dict

    result = rbodydict['class_name']

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : result
    }
