# Custom Object Detection with TensorFlow

![](screenshots/starwars_small.gif)

## Installation Instructions

First, with python and pip installed, install the scripts requirements:

```
pip install -r requirements.txt
```
Then you must compile the Protobuf libraries:

```
protoc object_detection/protos/*.proto --python_out=.
```

Add `models` and `models/slim` to your `PYTHONPATH`:

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

**IMPORTANT NOTE:** This must be ran every time you open terminal, or added to your `~/.bashrc` file.


## Creating TensorFlow Records
Run the script:

```
python object_detection/create_tf_record.py
```

Once the script finishes running, you will end up with a `train.record` and a `val.record` file. This is what we will use to train the model.

## Downloading a Base Model
Training an object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we’ll take an object detector trained on a different dataset, and reuse some of it’s parameters to initialize our new model.

You can download a model from this [model zoo](https://github.com/bourdakos1/Custom-Object-Detection/blob/master/object_detection/g3doc/detection_model_zoo.md). Each model varies in accuracy and speed. I used `faster_rcnn_resnet101_coco`.

Extract the files and move all the `model.ckpt` to our models directory.

You should see a file named `faster_rcnn_resnet101.config`. It’s set to work with the `faster_rcnn_resnet101_coco model`. If you used another model, you can find a corresponding config file [here](https://github.com/bourdakos1/Custom-Object-Detection/tree/master/object_detection/samples/configs).

## Train the Model
Run the following script and it should start to train!

```
python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config
```

**Note:** Replace `pipeline_config_path` with the location of your config file.

The model that I used in the video ran for about 22k steps. If you think running locally takes to long the next steps show how to train the model with PowerAI.

## PowerAI
PowerAI lets us train our model on IBM Power Systems.

It only took about an hour to train for 10k steps. However, this was just with 1 GPU! The real power in PowerAI comes from the the abilty to do distributed deep learning across hundreds of GPUs with up to 95% efficiency.

With the help of PowerAI, IBM just set a new image recognition record of 33.8% accuracy in 7 hours, surpassing the previous industry record set by Microsoft — 29.9% accuracy in 10 days.

For this project, I definitely didn’t need those kind of resources, because I am not training on millions of images. 1 GPU will do.

## Creating a Nimbix Account
Nimbix provides developers a trial account that provides 10 hours of free processing time on the PowerAI platform.

You can register [here](https://www.nimbix.net/cognitive-journey/).

**Note:** This process is not automated so it may take up to 24 hours to be reviewed and approved.

Once approved, you should receive an email with instructions on confirming and creating your account. It will ask you for a “Promotional Code”, but leave it blank.

You should now be able to log in [here](https://mc.jarvice.com).

## Deploy the PowerAI Notebooks Application
Start by searching for `PowerAI Notebooks`.

![](https://cdn-images-1.medium.com/max/1600/1*X41PZafFtX055NnbwBacEg.png)

Click on it and then choose `TensorFlow`.

![](https://cdn-images-1.medium.com/max/1600/1*rFh7QVFGs_QzELReRFyAxQ.png)

Choose the machine type of `32 thread POWER8, 128GB RAM, 1x P100 GPU w/NVLink (np8g1)`.

![](https://cdn-images-1.medium.com/max/1600/1*I0ycKwK54z2MdSbuma05vg.png)

Once started, the following dashboard panel will be displayed. When the server `Status` turns to `Processing`, the server is ready to be accessed.

Get the password by clicking on `(click to show)`.

Then, click `Click here to connect` to launch the Notebook.

![](https://cdn-images-1.medium.com/max/1600/1*JLWTTJT4rUmxLN69lKdFaA.png)

Log-in using the user name `nimbix` and the previously supplied password.

![](https://cdn-images-1.medium.com/max/1600/1*wXLlUuNvo_qPO-_p4kfjKA.png)

## Start Training
Get a new terminal window by clicking on the `New` pull-down and selecting `Terminal`.

![](https://cdn-images-1.medium.com/max/1600/1*j8z6DLJgjyvH13-KXfMajQ.png)

**Note:** Terminal may not work in Safari.

The steps for training are the same as they were when we ran this locally. If you’re using my training data then you can just clone my repo by running (If not, just clone your own repo):

```
git clone https://github.com/bourdakos1/Custom-Object-Detection.git
```

Then cd into the root directory:

```
cd Custom-Object-Detection
```

Then run this snippet, which will download the pretrained `faster_rcnn_resnet101_coco` model we downloaded earlier.

```
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
mv faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* .
```

Then we need to update our `PYTHONPATH` again, because this in a new terminal:

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Then we can finally run the training command again:

```
python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config
```

## Downloading Your Model
When is my model ready? It depends on your training data, the more data, the more steps you’ll need. My model was pretty solid at ~4.5k steps. Then, at about ~20k steps, it peaked. I even went on and trained it for 200k steps, but it didn’t get any better.

I recommend downloading your model every ~5k steps and evaluate, to make sure you’re on the right path.

Click on the `Jupyter` logo in the top left corner. Then, navigate the file tree to `Custom-Object-Detection/train`.

Download all the model.ckpt files with the highest number.
*. `model.ckpt-STEP_NUMBER.data-00000-of-00001`
*. `model.ckpt-STEP_NUMBER.index`
*. `model.ckpt-STEP_NUMBER.meta`

**Note:** You can only download one at a time.

![](https://cdn-images-1.medium.com/max/1600/1*2NUyMsF4SoVv1Jm0zMwc8Q.png)

**Note:** Be sure to click the red power button on your machine when finished. Otherwise, the clock will keep on ticking indefinitely.

## Export the Inference Graph
In order to use the model in our code we need to convert the checkpoint files (`model.ckpt-STEP_NUMBER.*`) into a frozen inference graph.

Move the checkpoint files you just downloaded, into the root folder of the repo you’ve been using.

Then run this command:

```
python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix model.ckpt-STEP_NUMBER \
        --output_directory output_inference_graph
```

_Remember ``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``_

You should see a new `output_inference_graph` directory with a `frozen_inference_graph.pb` file. This is the file we need.

## Test the Model
Just run the following command:

```
python object_detection/object_detection_runner.py
```

It will run your object detection model found at `output_inference_graph/frozen_inference_graph.pb` on all the images in the `test_images` directory and output the results in the `output/test_images` directory.

## Results
Here’s what we get when we run our model over all the frames in this clip from Star Wars: The Force Awakens.

[![Watch the video](screenshots/youtube.png)](https://www.youtube.com/watch?v=xW2hpkoaIiM)

