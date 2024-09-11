# Pest sensor using Edge Impulse with synthetic training data created with NVIDIA Omniverse Replicator

![](img/capture_carpet.png "Synthetic image generation")

**Synthetic image generation, 2 images pr. second, render Eivind Holt**

## Intro
This article describes how we can build an object detection model for visual detection of unwanted insects. The model can run on low-cost, low-energy hardware and can therefore be placed in many rooms around a building. Compared to the traditional use of sticky cardboard traps this poses several advantages:
* Can be trained to detect and ignore any kind of species
* Images of the detected specimen can instantly be inspected on phone by notification
* Reusable once the situation has been handled, unlimited lifespan
* Can be combined with replenishable bait

These are huge advantages especially for large building management, agriculture and insurance claims, where checking each physical trap can be impractical.

This article will focus on building object detection models using synthetically generated datasets. 

* Using NVIDIA Omniverse to generate pre-labeled images
* How many images do we need?
* Effects of changing model parameters
* Regenerating dataset on hardware changes, optics
* Testing models on 3D-printed objects
* The future of synthetic datasets

## Hardware used
* NVIDIA GeForce RTX 3090 (any RTX will do)
* [OpenMV Cam RT1062](https://openmv.io/products/openmv-cam-rt)
    * Ultra Wide Angle Lens
* NVIDIA Jetson Orin Nano 8 GB
* FormLabs Form 3

## Software used:
* [Edge Impulse Studio](https://studio.edgeimpulse.com/studio)
* [NVIDIA Omniverse Code](https://www.nvidia.com/en-us/omniverse/) with [Replicator](https://developer.nvidia.com/omniverse/replicator)
* [Visual Studio Code](https://code.visualstudio.com/)
* [Blender](https://www.blender.org/)
* [Autodesk Fusion 360](https://www.autodesk.no/products/fusion-360/)

Project [Impulse](https://studio.edgeimpulse.com/public/322153/latest) and [code repository](https://github.com/eivholt/pest-control-synth).
Datasets [Kaggle]

[The Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) tells us that a neural network with enough complexity (neurons, layers, connections) can approximate any continuous function to a desired degree of accuracy. It doesn’t directly address the trade-off between the complexity of the network and the amount of training data. Any implemented machine learning model will suffer from practically imposed limitations. With TinyML this is put to the extreme. Many sacrifices are made to be able to run real-time inference on a microcontroller with limited memory space - neural network layer depth, color information, image resolution, to name a few. However, the amount of training data used does not affect the size or execution time of the resulting model. While the size of the dataset can't compensate for a neural network that is too simple to fully capture the complexity of the underlying task, we will see how it can play a great role in avoiding overfitting. In other words, we will see how we can take full advantage of our limited neural network by providing the best possible pre-labeled training and testing images.

## NVIDIA Omniverse Replicator
To generate synthetic object images that come complete with labels we can create a 3D scene in NVIDIA Omniverse and use it's Replicator Synthetic Data Generation toolbox. With this we are able to create thousands of slightly varying images, a concept called domain randomization. With a NVIDIA RTX 3090 graphics card from 2020 it is possible to produce about one path-traced images per second. 20 000 images would take about 4 hours to create. In a lack of a sufficient local GPU, 3D scene setup can be created remotely and dataset generation can be offloaded, headlessly, to a cloud based GPU farm.

Please refer to the following projects for more details on working with NVIDIA Omniverse Replicator:

[Reflective surfaces](https://docs.edgeimpulse.com/experts/featured-machine-learning-projects/surgery-inventory-synthetic-data)
![](img/surgery_example.png "Surgery inventory project")

[Transparent materials](https://docs.edgeimpulse.com/experts/featured-machine-learning-projects/rooftop-ice-synthetic-data-omniverse)
![](img/icicle-still01.png "Icicle project")

For this project a number of species of insects were selected to test the concept. Silverfish and cockroaches represent insects that are unwanted indoors, while ants and flies can be ignored. Already a choice has to be made - will we only label and train our model to recognize silverfish and cockroaches? In that case ants and flies can be present in the images and the model will learn to ignore them as background. For this test however the model was trained to the extreme - to identify all four classes. Keep this in mind when we look at the model accuracy. For a real-word application this is unnecessarily complex and results in penalties in accuracy.

High-fidelity 3D models were aquired and prepared for import in Omniverse. Due to licensing restrictions the models are not shared in the project repo.

<table>
  <tr>
    <td>
      <a href="img/silverfish_collection_thumbnail_07.jpg">
        <img src="img/silverfish_collection_thumbnail_07.jpg" alt="Silverfish" width="500" />
      </a>
    </td>
    <td>
      <a href="img/cockroach_poses_collection_001.jpg">
        <img src="img/cockroach_poses_collection_001.jpg" alt="Cockroach" width="500" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="img/ant_05_thumbnail_0001.jpg">
        <img src="img/ant_05_thumbnail_0001.jpg" alt="Ant" width="500" />
      </a>
    </td>
    <td>
      <a href="image4_fullsize.jpg">
        <img src="img/house_fly_standing_thumbnail_0001.jpg" alt="Housefly" width="500" />
      </a>
    </td>
  </tr>
</table>

**Selection of insects, render Tornado Studio**

## Converting 3D models
3D models come in all kinds of formats, unfortunately the USD format that Omniverse is heavily invested in has not yet overrun model market places. Converting both 3D geometry, materials, rigging and animation can be tricky, at best. Upon request market places like TurboSquid will strive to convert models to your needs, with varying success. In this project all of the models were incomplete. Careful examination of the missing reference names, some discussion with ChatGPT, some guesswork and a lot of experimentation produced useful results.

NVIDIA provides a branch of Blender where work is being done to improve USD export.

![](img/Blender_USD_branch.png "Blender USD branch")

**Blender USD branch, screenshot Eivind Holt**

When materials on imported models appear incomplete, check File>External Data>Report Missing Files and Find Missing files. Sometimes there is just a problem with paths, other times files are actually missing. In this project it gave an indication what types of material components were missing. Material type Principled BSDF is recommended for export to Omniverse.

![](img/Blender_material.png "Blender material")

**Blender materials, screenshot Eivind Holt**

## Domain randomization
With Replicator we can create a program that randomizes any aspect of the 3D scene we want. In this project the camera was randomly placed along a plane above the insects. 

![](img/camera_plane.png "Random camera placement")

**Random camera placement, screenshot Eivind Holt**

The insects were randomly scattered on an invisible plane within the confides of the floor. Light sources were randomly enabled. Inspect the [source code](https://github.com/eivholt/pest-control-synth/blob/main/Omniverse/replicator_init.py) for completeness.

```python
with rep.new_layer():
	def scatter_insects(insects, scatter_plane):
		with insects:
			rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (0, 360, 0)))
			rep.randomizer.scatter_2d(surface_prims=scatter_plane, check_for_collisions=True)
		return insects.node
	
	def randomize_camera(insects, camera_plane):
		with camera:
			rep.randomizer.scatter_2d(surface_prims=camera_plane, check_for_collisions=False)
			rep.modify.pose(look_at=insects)
		return camera.node
	
	def randomize_spotlights(lights):
		with lights:
			rep.modify.visibility(rep.distribution.choice([True, False]))
		return lights.node

	def randomize_floor(floor, materials):
		with floor:
			# Load a random scene material file as texture
			rep.randomizer.materials(materials)
		return floor.node
```

An invisible cylinder contained each insect to help keep a bit of distance between them, in effect a _bounding box_. This is useful as we place many insects on each image and FOMO will merge detected object centroids that are close to each other. In reality this sensor is probably not useful in situations where lots of insects overlap. It will be of little importance if it counts 99 cockroaches instead of 100.

![](img/bounding_box.png "Bounding box")

**Boundaries, please! Screenshot Eivind Holt**

![](img/bounding_box_top.png "Bounding box")

**Bounding boxes (cylinders) to avoid overlap, screenshot Eivind Holt**

In half of the generated images we randomly use realistically looking floor materials. The materials are of high quality and provided in the Omniverse Material catalogue.

![](img/Omniverse_materials.png "Omniverse Materials")

![](img/output_1.png "Images with random floor materials")

**Images with random floor materials, screenshot Eivind Holt**

In the other half of the images we try to combat a challenge particularly present in this application. By the nature of the features of the insects we are looking for, basically dark blobs, a finished model running on low resolution has problems with false positives when exposed to dark objects in low light. This can be mitigated simply by placing the device in well lit areas without the presence lots of clutter. False positives were minimized by creating half of the training samples on top of random images. The material of the floor was simply changed to a random image from an open dataset, [COCO 2017](https://cocodataset.org/#home). This method reduces model accuracy during training, but model validation and real-life testing proves this increases robustness.

```python
def randomize_floor_coco(floor, texture_files):
		with floor:
			# Load a random .jpg file as texture
			rep.randomizer.texture(textures=texture_files)
		return floor.node

folder_path = '[path-to-coco]/val2017/testing/'
texture_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

with rep.trigger.on_frame(num_frames=20000, rt_subframes=20):
    rep.randomizer.randomize_floor_coco(floor, texture_files)
```
![](img/output_2.png "Images with random backgrounds")

**Images with random backgrounds, nightmare fuel, screenshot Eivind Holt**

Included in the project repository is a [script for creating a labels-file](https://github.com/eivholt/pest-control-synth/blob/main/Omniverse/basic_writer_to_pascal_voc.py) compatible with Edge Impulse ingestion API.

Note: Replicator produces artifacts in images below 256*256 pixel resolution. Edge Impulse takes care of resizing training images that are larger than model input size and at this level any artifacts caused by interlacing does not seem to significantly affect the results.

## Dataset size
When we are no longer limited by how many labeled images we can generate, how can we determine just how many we need? In short; it depends. It's impossible to provide an universal number of formula, but experimentation builds intuition. An object detection model that classifies elephants from tennis balls probably only needs a small number of images. Would love to see _that_ application, by the way. Distinguishing insects with limited hardware is quite challenging, but a great dataset can help model training generalize and avoid overfitting. The following examples and numbers are spesific to this application and can not be directly transferred to others, however the approach can.

### Number of images
By containing a number of variables we can apply the following method to get a feeling how the size of the dataset affects accuracy. In this experiment pretrained FOMO MobileNetV2 0.35 was used with grayscale images at 256*256 pixels. No data augmentation was used and the training F1-score represents the int8 models, while the model tested on a separate validation set (the percentage of all samples with precision score above 80%) is the float32 variant. The difference between the two precision levels are consistent and negligable. The datasets were testet for 120 and 240 epochs.

![](img/Sample%20size.svg "Sample size significance")

**Sample size significance, graph by Eivind Holt**

| _Samples_ | Model F1 (120 epochs) | _Test F1 (120 epochs)_ | Model F1 (240 epochs) | _Test F1 (240 epochs)_ |
|----------------------------|----------------------|----------------------|-----------------------|----------------------|
| 200                        | 72.60                | 32.50                | 74.20                 | 42.50                |
| 2,000                      | 83.30                | 72.14                | 84.00                 | 71.64                |
| 6,000                      | 86.20                | 83.21                | 87.20                 | 85.62                |
| 18,000                     | 87.60                | 84.54                | 87.90                 | 84.35                |
| 42,000                     | 89.90                | 89.38                | 90.50                 | 90.14                |

**Sample size significance, table by Eivind Holt**

From this we can visualize at what rate accuracy increases with dataset size. We can see how accuracy flattens as the dataset grows relatively large, and decide when we think adding more images will have diminishing effect. Required compute power and time increases with training samples, we want to find a sweet spot. We can also see how the effects of the number of training epochs plays less of a role when the dataset grows.

Observe how for _this application_ at _this resolution_ we would need to manually capture and label more than 6,000 images, requiring more than 70,000 labels, to achieve remotely useful results!

## Testing model in virtual environment
Before we start digging for insects to test our new model on it's handy to test it in a virtual environment. Edge Impulse supplies [an extension](https://github.com/edgeimpulse/edge-impulse-omniverse-ext) both for uploading training data and for testing models. Once installed, making sure our project has a WebAssembly built, we can load the same USD-scene used for generating images in Isaac Sim and perform classification.

![](img/IsaacSim.png "Isaac Sim")

**NVIDIA Isaac Sim, screenshot by Eivind Holt**

![](img/EI_wasm.png "wasm")

**WebAssembly deployment, screenshot by Eivind Holt**

![](img/IsaacSim_EI.png "Isaac Sim Edge Impulse classification")

**NVIDIA Isaac Sim, top left: rendered image, bottom left: ground truth, bottom righ: model prediction, screenshot by Eivind Holt**

![](img/IsaacSim_EI_2.png "Isaac Sim Edge Impulse classification")

**NVIDIA Isaac Sim, top left: rendered image, bottom left: ground truth, bottom righ: model prediction, screenshot by Eivind Holt**

For running on [OpenMV Cam RT1062](https://openmv.io/products/openmv-cam-rt) a model with 256
(OpenMV Cam RT1062 photo)

![](img/42k_openmv_training.png "FOMO 42k images training")

**FOMO object detection model training, screenshot by Eivind Holt**

![](img/42k_openmv_test.png "FOMO 42k images model testing")

**FOMO object detection model testing, screenshot by Eivind Holt**

## Regenerating dataset on hardware changes, optics
With traditionally captured an labeled images we should use hardware and optics as close to the intended device where inference is supposed to be performed. Because of the labor required with manually captured training data, we are practically stuck with those choices. But what if new and better options appear? What if we discover our initial choices were misguided?

### Change of optics
In this project we demonstrate the versability of synthetically generated images with two examples.
First, we want to test if changing the optics of our device to a wide angle (fisheye) can improve the area covered per device. Had the dataset been manually captured and labeled, our labor investment would probably force us to make some method for distorting the images and labels to approximate the intended lens. If this produced unsatisfactory results there would be no alternative than to scrap the dataset and start over.

![](img/wideangle_omniverse.png "Wide-angle lens")

**Wide-angle lens, screenshot by Eivind Holt**

![](img/wideangle_dataset.png "Wide-angle dataset regeneration")

**Wide-angle dataset regeneration, screenshot by Eivind Holt**

In Omniverse we can simply change the desired parameters of the virtual camera and regenerate all the images. For better results the simulated optics should more closely be matched the physical counterpart. Also, there may be a need for more training images, as the complexity of the task has increased.

![](img/40k_fomo_WA_240epochs.png "Wide-angle model training")

**Wide-angle model training, screenshot by Eivind Holt**

![](img/40k_fomo_WA_240epochs_bw_test.png "Wide-angle model testing")

**Wide-angle model testing, screenshot by Eivind Holt**

### Change inference hardware
Second, we want to test more capable hardware, sacrificing portability and cost for higher accuracy and frame rate. Had our images been captured at a resolution close to the capability of the originally intended end-device, there would be no accurate way to increase resolution. We could cheat with AI upscaling, but we could not guarantee the quality of the most important aspects of increasing resolution - the details. With synthetically generated images we can just change the output resolution and regenerate.
(Jetson Nano)

In the following graph we have created training samples at 640*640px and compared accuracy of neural networks with increasing number of samples (split 80/20 training/testing). Neural network input is grayscale, data augmentation is disabled and number of epochs are set to 240.

![](img/Training%20sample%20size.svg "Sample number significance")

**Sample number significance, graph by Eivind Holt**

| Samples | Model F1 | Test F1 |
|---------|----------|---------|
| 200     | 88.70    | 81.63   |
| 2000    | 96.00    | 97.78   |
| 6000    | 97.20    | 99.28   |
| 12000   | 98.00    | 99.79   |
| 18000   | 98.20    | 99.75   |

**Sample number significance, table by Eivind Holt**

At 640*640px resolution our neural network is picking up important details to separate the insects already at 200 samples, where 81.63% of all samples with testing precision score above 80%. Still, a dramatic increase in accuracy can be witnessed by increasing to 2,000 samples, 97.78%. Manually capturing 200 images and labeling 2,400 objects is a massive one-person undertaking. I would not consider doing this for 2,000 images with 24,000 labels. Further increasing the number of samples we observe how this particular model tangents full accuracy.

![](img/18k_640_training.png "FOMO 640*640 px 18,000 samples training")

**FOMO 640*640 px 18,000 samples training, screenshot by Eivind Holt**

![](img/18k_640_testing.png "FOMO 640*640 px 18,000 samples testing")

**FOMO 640*640 px 18,000 samples testing, screenshot by Eivind Holt**

### Myth #1: Increasing model resolution without increasing training sample resolution does not increase accuracy
Intuitively one might think that there is no point in increasing the input resolution of the neural network above the resolution of the samples we are training and testing on. In the following test 42,000 images of resolution 256*256 pixels were used to demonstrate how accuracy changes when only model input resolution changes. Other hyper parameters were identical, only grayscale image info was used, epochs at 120.

![](img/Model%20input%20resolution.svg "Model resolution surpassing sample resolution")

**Model resolution surpassing sample resolution, graph by Eivind Holt**

| Resolution | Model F1 (120 epochs) | Test F1 (120 epochs) |
|------------|-----------------------|----------------------|
| _640_      | 95                    | 96.57                |
| _320_      | 91.8                  | 92.9                 |
| 256        | 89.9                  | 89.38                |
| 160        | 83.3                  | 72.51                |
| 96         | 71.6                  | 36.69                |

**Model resolution surpassing sample resolution, graph by Eivind Holt**

Model testing accuracy increases when neural network input resolution surpasses input sample resulution (256*256 px). With the capability of effortlessly generating higher resolution training samples, there is no reason for not doing so when increasing neural network resolution. However, this test is useful if we are limited by costly manual data preparation. Would we trust these results if we only had a limited amount of manually captured and labelled data?

## Jetson inference
After [flashing NVIDIA Jetson Orin Nano with JetPack 6.0](https://www.jetson-ai-lab.com/initial_setup_jon.html) we can either run our model natively by the [CLI edge-impulse-linux-runner](https://docs.edgeimpulse.com/docs/run-inference/linux-eim-executable). Note: At the time of writing we have to supply --force-target runner-linux-aarch64-jetson-orin-6-0 to correctly target CUDA 12.0.

Alternatively we can download and run a [container image as described here](https://docs.edgeimpulse.com/docs/run-inference/docker).

Please find a simple [python program for running continous inference on web cam feed](https://github.com/eivholt/pest-control-synth/blob/main/Jetson/pest-continous-webcam.py).

## Future of synthetic data

* EI can generate images provided model and script and determine how many images are needed.


[![Chrome objects detection demo](https://img.youtube.com/vi/1k0pfPwzTw4/0.jpg)](https://www.youtube.com/watch?v=1k0pfPwzTw4)

**Chrome objects detection demo**




## Creating label file for Edge Impulse Studio
Edge Impulse Studio supports a wide range of image labeling formats for object detection. Unfortunately the output from Replicator's BasicWriter needs to be transformed so it can be uploaded either through the web interface or via [web-API](https://docs.edgeimpulse.com/reference/ingestion-api#ingestion-api).

Provided is a simple Python program, [basic_writer_to_pascal_voc.py](https://github.com/eivholt/surgery-inventory-synthetic-data/blob/main/omniverse-replicator/basic_writer_to_pascal_voc.py). A simple prompt was written for ChatGPT describing the output from Replicator and the [desired results described at EI](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/uploader#understanding-image-dataset-annotation-formats). Run the program from shell with
``` 
python basic_writer_to_pascal_voc.py <input_folder>
```
or debug from Visual Studio Code by setting input folder in launch.json like this: 
```
"args": ["../out"]
```

This will create a file bounding_boxes.labels that contains all labels and bounding boxes per image.



## Results
The results of this project show that training and testing data for object detection models can be synthesized using 3D models, reducing manual labor in capturing images and annotation. Even more impressive is being able to detect unpredictable reflective surfaces on heavily constrained hardware by creating a large number of images.

## Conclusion
The domain of visual object detection is currently experiencing a thrilling phase of evolution, thanks to the convergence of numerous significant advancements. Envision a service capable of accepting 3D models as input and generating a diverse array of images for training purposes. With the continuous improvements in generative diffusion models, particularly in the realm of text-to-3D conversion, we are on the cusp of unlocking even more potent capabilities for creating synthetic training data. This progression is not just a technological leap; it's set to revolutionize the way we approach object detection, paving the way for a new generation of highly innovative and effective object detection solutions. The implications of these advancements are vast, opening doors to unprecedented levels of accuracy and efficiency in various applications.

## Disclosure
I work with research and innovation at [DIPS AS](https://www.dips.com/), exploring the future of medical technology. I am a member of Edge Impulse Expert Network. This project was made on my own accord and the views are my own.
