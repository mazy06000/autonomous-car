import json
import streamlit as st
import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import cv2
import streamlit.components.v1 as components
import webbrowser

st.set_page_config(layout='centered')

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

# st.markdown("""
#                 <style>
#                 .centered-container {
#                 background-color: #FFFFFF;
#                 display: inline-flex;
#                 padding: 5px;
#                 }
#                 .link {
#                 color: back;
#                 cursor: pointer;
#                 font-weight: 400;
#                 text-decoration: none;
#                 }
#                 .link--arrowed {
#                 display: inline-block;
#                 height: 2rem;
#                 line-height: 2rem;
#                 }
#                 .link--arrowed .arrow-icon {
#                 position: relative;
#                 top: -1px;
#                 -webkit-transition: -webkit-transform 0.3s ease;
#                 transition: -webkit-transform 0.3s ease;
#                 transition: transform 0.3s ease;
#                 transition: transform 0.3s ease, -webkit-transform 0.3s ease;
#                 vertical-align: middle;
#                     transform: rotate(180deg);
#                 }
#                 .link--arrowed .arrow-icon--circle {
#                 transition: stroke-dashoffset 0.3s ease;
#                 stroke-dasharray: 95;
#                 stroke-dashoffset: 95;
#                 }
#                 .link--arrowed:hover .arrow-icon {
#                 transform: translate3d(5px, 0, 0);
#                     transform: rotate(180deg);
#                 }
#                 .link--arrowed:hover .arrow-icon--circle {
#                 stroke-dashoffset: 0;
#                 }

#                 a:link { text-decoration: none; }
#                 a:visited { text-decoration: none; }
#                 a:hover { text-decoration: none; }
#                 a:active { text-decoration: none; }
#                 .css-znku1x a {
#                     color: black
#                 }
#                 </style>
                
#                 <div style="display:flex;justify-content: center; margin-bottom: 25px;"">
#                     <section class="centered-container">
#                     <a class="link link--arrowed" onclick="backToProtfolio(event)" href="https://www.mohamed-mazy.com">
#                     <svg class="arrow-icon" xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
#                         <g fill="none" stroke="black" stroke-width="1.5" stroke-linejoin="round" stroke-miterlimit="10">
#                             <circle class="arrow-icon--circle" cx="16" cy="16" r="15.12"></circle>
#                             <path class="arrow-icon--arrow" d="M16.14 9.93L22.21 16l-6.07 6.07M8.23 16h13.98"></path>
#                         </g>
#                     </svg>
#                     Back to portfolio
#                     </a>
#                     </section>
#                 </div>
#                 """, unsafe_allow_html=True)


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
    left.markdown('<div><b>Dataset:</b> ImageNet</div>',
                  unsafe_allow_html=True)

    right.subheader("Model performance")
    right.markdown('<div><b>Test IoU:</b> 63,6%</div>', unsafe_allow_html=True)


# with credit:
#     st.markdown("""<div style="text-align: center; margin-top: 25px;"">
#     By <button onClick='window.location.href="https://www.mohamed-mazy.com"'>Mohamed Mazy</button></div>""", unsafe_allow_html=True)
