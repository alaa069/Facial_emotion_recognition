# Face Emotion Recognition Project
**Face emotion recognition using convolutional neural networks**

Aim of this project was to create a simple library to recognize one of 7 emotions (*anger, contempt, disgust, fear, happy, sadness, surprise*) of the person on the image using either pre-trained network by us or train it by yourself firstly.

# Run server
To run the server it's suggested to use [gunicorn](http://gunicorn.org/).
```bash
gunicorn server:app
```

And later navigate to [127.0.0.1:8000](http://127.0.0.1:8000) (default port, you can specify another).


[Dataset](http://www.consortium.ri.cmu.edu/ckagree/)

As base algorithm for this project Convolutional Neural Networks have been used (modification of LeNet architecture, with *dropout* technique).

## Using
In order to use pre-trained network you can load saved network
```python
network = nt.build_cnn('models/model.npz')
```
load image
```python
faces = net_training.load_img("images/anger.jpg")
```
and evaluate this image with a given network
```python
tab = net_training.evaluate(network, faces)
```

Optionally, if you want to train network by yourself you need to create a certain folder structure:

 - Data
   * Images
     - 1
       - photo1.jpg  
        - photo2.jpg
     - 2
       - photo1.jpg
        - ...
     - ...
   * Labels
     - 1
       - label.txt
     - 2
       - label.txt
     - ...
   
     
and execute function to train by providing specified paths
```python
net_training.train_net(datadir, imagedir, labeldir, network)
```

## Examples
![Anger](/images/anger.jpg?raw=true)
```
anger = 100.00%
contempt = 0.00%
disgust = 0.00%
fear = 0.00%
happy = 0.00%
sadness = 0.00%
surprise = 0.00%
```

![Disgust](/images/disgust.jpg?raw=true)
```
anger = 0.49%
contempt = 0.02%
disgust = 97.62%
fear = 1.47%
happy = 0.00%
sadness = 0.40%
surprise = 0.00%
```

![Surprise](/images/surprise.jpg?raw=true)
```
anger = 0.00%
contempt = 0.00%
disgust = 0.00%
fear = 0.00%
happy = 0.00%
sadness = 0.08%
surprise = 99.92%
```

![Sadness](/images/sadness.jpg?raw=true)
```
anger = 0.00%
contempt = 0.00%
disgust = 14.98%
fear = 0.00%
happy = 0.00%
sadness = 85.02%
surprise = 0.00%
```
