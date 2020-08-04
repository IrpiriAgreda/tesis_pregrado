//NN with ONNX
// const onnx = require('onnxjs');
//
// //Front end with bokeh
// const bokeh = require('bokehjs');

//Express
const express = require('express');
const path = require('path')
const app = express();

//app.use(express.static(path.join(__dirname, '/thesis_js/')));

//Get root
app.get('/', function(req, res){
  res.sendFile('index.html', { root: __dirname });
});

//Serve static
app.get('/plot.js', function(req,res){
    res.sendFile(path.join(__dirname + '/plot.js'));
});

app.get('/inputs/actor_critic.onnx', function(req,res){
    res.sendFile(path.join(__dirname + '/inputs/actor_critic.onnx')); 
});

//Listen
app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});
