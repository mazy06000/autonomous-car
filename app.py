import json
import streamlit as st
import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import cv2
import streamlit.components.v1 as components

if "model" not in st.session_state:
    model = tf.keras.models.load_model("models/segment_self_driving.h5", custom_objects={'jaccard_loss':sm.losses.JaccardLoss(),
                                                   'iou_score':sm.metrics.IOUScore(threshold=0.5)})
    st.session_state["model"] = model

catid2color = {0: [0, 0, 0],
                   1: [128, 64, 128],
                   2: [70, 70, 70],
                   3: [153, 153, 153],
                   4: [107, 142, 35],
                   5: [70, 130, 180],
                   6: [220, 20, 60],
                   7: [0, 0, 142]}
max_len = 55

header = st.container()
uploading = st.container()
characteristic = st.container()

with header:
    st.title('AUTONOMOUS CAR SEGMENTATION')

with uploading:

    image_uploaded = st.file_uploader("Choose a street image")
    if image_uploaded is not None:
        left, right = st.columns(2)
        nparr = np.fromstring(image_uploaded.read(), np.uint8)
        img_to_predict = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_to_predict = cv2.resize(img_to_predict, (256,128))
        left.write("Original image")
        left.image(img_to_predict, channels="BGR", width=350)

        # make prediction
        predicted_img = np.squeeze(np.argmax(st.session_state["model"].predict(np.expand_dims(img_to_predict, 0)), axis=-1))

        # build the mask
        pred_colors = np.empty(predicted_img.shape + (3,), dtype='uint8')
        for row in range(predicted_img.shape[0]):
            for col in range(predicted_img.shape[1]):
                pred_colors[row, col] = catid2color[predicted_img[row, col]]
        right.write("Image predicted")
        right.image(pred_colors, width=350)

        components.html('''

        <style>
        
        .selected-items {
            border: 2px solid #ff0000;
            line-height: 32px;
        }

        .my-legend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
        }

        .my-legend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
        }

        .my-legend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            margin-left: 0;
            line-height: 18px;
            margin-bottom: 2px;
        }

        .my-legend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 1px solid #999;
        }

        .my-legend .legend-source {
            font-size: 70%;
            color: #999;
            clear: both;
        }

        .my-legend a {
            color: #777;
        }

        .legend-labels {
            display: flex;
            width: 100%;
            justify-content: space-between;
        }
        
        </style>

        <div class='my-legend'>
        <div class='legend-title'>Segments</div>
            <div class='legend-scale'>
                <ul class='legend-labels'>

                    <li><span style='background:#000000;'></span>void</li>
                    <li><span style='background:#804080;'></span>flat</li>
                    <li><span style='background:#464646;'></span>construction</li>
                    <li><span style='background:#999999;'></span>object</li>
                    <li><span style='background:#6B8E23;'></span>nature</li>
                    <li><span style='background:#4682B4;'></span>sky</li>
                    <li><span style='background:#DC143C;'></span>human</li>
                    <li><span style='background:#00008E;'></span>vehicle</li>
                </ul>
            </div>
        </div>
        </div>


        
        ''')

with characteristic:
    st.header("Characteristic")
    left, right = st.columns(2)
    left.subheader("About Model")
    left.markdown('<div><b>Model:</b> Unet-MobileNet </div>',
                  unsafe_allow_html=True)
    left.markdown('<div><b>Dataset:</b> ImageNet (uncased)</div>',
                  unsafe_allow_html=True)

    right.subheader("Model performance")
    right.markdown('<div><b>Test IoU:</b> 63,6%</div>', unsafe_allow_html=True)
