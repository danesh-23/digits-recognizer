# Digits Recognizer :hash:

## What is it?
This is a simple machine learning program that learns how to identify digits from 0-9 by analysing real human handwriting provided through the MNIST dataset.

## Getting Started
1. Download the file [here](https://github.com/danesh-23/digits-recognizer/archive/master.zip) as a zip or git clone it using the command below.
```
  git clone https://github.com/danesh-23/digits-recognizer.git
```

2. You need to have a few external libraries installed in your IDE, including keras and matplotlib. You can download keras from their main page [here](https://pypi.org/project/Keras/#files) and matplotlib [here](https://pypi.org/project/matplotlib/#files).  
Alternatively, you can install them within your IDE's or through the commandline by copy-pasting the following commands.
```
  pip install Keras
 ```
 ```
  pip install matplotlib
```

2. Run the digits-recognizer.py file in any IDE such as Pycharm, Eclipse, Atom etc. and you can feed it any handwritten digit taken from the MNIST dataset.

3. The machine learning model will take between 15 minutes up to an hour to train on the MNIST dataset of 60000 images and you can see how it performed as well as some of the images it may have not been able to learn and predict successfully.

Thats it for now. Try improving it to predict on your own handwritten digits :thinking:

## A little about this project

I started working on this project further along my journey of basic machine learning, on my way to learning deep neural networks. So, I decided to try the classic detect MNIST digits challenge which arguably every data scientist has worked on at some point of learning neural networks. I have many ideas to expand this project such as by allowing it to accept images outside of the MNIST dataset and give the user more options in their choice of images and also to improve the current accuracy rate of 99.1% by fine tuning some hyper-parameters and adding/removing layers to hopefully get it up to 99.5-6 :crossed_fingers:

## Reporting Bugs

To report a bug, you may use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)

## Feature Request

If you have any ideas of interesting features you would like added, you may fill the [Feature Request form](.github/ISSUE_TEMPLATE/feature_request.md)

## Project maintainers

This project is maintained by Danesh Rajasolan(me). Use of this project under the [MIT License](LICENSE.md).
