import logo from './logo.svg';
import './App.css';

import React, { useState, useEffect, Fragment } from "react";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null)
  const videoRef = React.useRef();
  const requestRef = React.useRef()
  const classRef = React.useRef("")

  useEffect(() => {
    setInterval(() => {
      detectFrame(videoRef.current, model);
    }, 100)
  }, [classLabels])

  const detectFrame = async (video, model) => {
    if (video && model) {
      let tfimg = tf.browser.fromPixels(video , 3 );
      let cast = tfimg.cast('float32');
      let reshaped = cast.reshape([1, tfimg.shape[0], tfimg.shape[1], tfimg.shape[2]]);
      let prediction = await model.predict(reshaped);
      let classScore = await prediction.data();
      let maxScoreId =  classScore.indexOf(Math.max(...classScore));
      
      if (classLabels) {
        classRef.current.innerHTML = `Prediction: ${classLabels[maxScoreId]}`
      }

      tf.dispose([tfimg, cast, reshaped, prediction])
    }
  }

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "environment"
          }
        })
        .then((stream) => {
          window.stream = stream;
          videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });
      
        const model_url = "tfjs/catdog/model.json";
        const modelPromise = tf.loadGraphModel(model_url);

        const classesPromise = fetch("tfjs/catdog/classes.json").then(response => { return response.json() });
        
        Promise.all([webCamPromise, modelPromise, classesPromise]).then(values => {
          const classes = Object.keys(values[2]);
          setClassLabels([classes[0], classes[1]])
          setModel(values[1])
        })
    }
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <video
            style={{ height: "600px", width: "500px" }}
            className="size"
            autoPlay
            playsInline
            muted
            ref={videoRef}
            width="600"
            height="500"
            id="frame"
          />

        <span ref={classRef}></span>
      </header>
    </div>
  );
}

export default App;
