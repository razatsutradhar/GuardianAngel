import base64
import io
import time

import cv2
import easyocr
# import matplotlib.image as mpimg
import imutils
import mysql.connector
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt


def report(liscense, lat, long):
    amberdb = mysql.connector.connect(host='sql5.freesqldatabase.com', user='sql5473936', passwd='43NA67P5Aw', database='sql5473936')
    mycurser = amberdb.cursor()
    sql = 'INSERT INTO Reports (Liscense, Latitude, Longitude, Time) VALUES (%s, %s, %s, %s)'
    val = (str(liscense), str(lat), str(long), str(time.time()))
    mycurser.execute(sql, val)
    amberdb.commit()

def contourIntersect(original_image, contour1, contour2):
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()

def get_plates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    locations = []
    addedContours = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)  # number here is how good the approximation is, higher = closer to edge ditection, lower = straight polygons
        if len(approx) == 4:
            locations.append(approx)
            addedContours.append(contour)


    print(locations)

    possiblePlates = []
    # possibleContours = []
    for i in range(0, len(locations)):
        location = locations[i]
        mask = np.zeros(gray.shape, np.uint8)
        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped = gray[x1:x2 + 1, y1:y2 + 1]
        process(cropped)
        possiblePlates.append([cropped, location])
    plt.show()
    return possiblePlates

#
def process(image) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def read_text(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    return result


from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents, filename, date):
    b64 = contents[contents.index('base64,')+7:]
    print(contents[0:40])
    print(contents[len(contents)-40:])
    base64_decoded = base64.b64decode(b64)

    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)

    plates = get_plates(image_np)

    allNumbers = []

    for crop in plates:
        plate = read_text(crop[0])
        for data in plate:
            allNumbers.append([data, crop[1]])
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    with open('./cache.txt') as f:
        flagged = f.readlines()
    f.close()

    flagged_stripped = [f.rstrip('\n') for f in flagged]

    print(allNumbers)
    tags = [result[0][1].replace(" ", "") for result in allNumbers]
    tags = list(set(tags))
    print(tags)



    retval, buffer = cv2.imencode('.jpg', image_np)
    jpg_as_text = contents[0:contents.index('base64,')+11] + str(base64.b64encode(buffer))[6:len(str(base64.b64encode(buffer)))-1]
    # print(flagged_stripped)

    triggerTag = None
    for tag in tags:
        if tag in flagged_stripped:
            triggerTag = tag
            report(tag, str(1.0), str(1.0))
            return html.Div([
                # HTML images accept base64 encoded strings in the same format
                # that is supplied by the upload
                html.H3("DANGER, tag " + str(triggerTag) + " is wanted\nLocation is being sent to authorities"),
                html.Img(src=jpg_as_text),

                html.Hr(),
                html.Div('Raw Content'),
                html.Pre(contents[0:200] + '...', style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all'
                })
            ])

    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=jpg_as_text),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True, host='democv.eastus.cloudapp.azure.com')



