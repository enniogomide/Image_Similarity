# **Image Similarity**

## **Description**

is an example of a routine to search for similar images based on a given image (photo). Features used especially on shopping sites to make suggestions.


## **Comments**

the routine uses a pre-trained network with weights for a huge range of photos. Based on others, additional images should be trained using the transfer learning feature.

For execution, the streamlit library was used, which is simple and allows local execution. 

The training transfer feature was not used because it could not be run on COLAB, which was unavailable several times.

Nor was it done on the notebook due to the unavailability of resources to run the training.

It took more than an hour and a half just to create the vectors.

It can be run on a laptop, executing each step in turn. If you want to use it in COLAB, you'll need to get the folder identification right.

The routine was put together using the material provided in the course, notebook suggestion:
 https://colab.research.google.com/github/sparsh-ai/rec-tutorials/blob/master/_notebooks/2021-04-27-image-similarity-recommendations.ipynb 
