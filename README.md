# Hybrid_Image
In this small project we try to implement a hybrid image based on `Img1.jpg` and `Img2.jpg` [1].

How to use
----------
For the sake of warping `Img2.jpg` on `Img1.jpg`, the user need to chose corresponding points between two images that are shown automatically on the screen. Moreover, the corresponding points should be chosen in a particular order. The image below shows the corresponding points chosen by user:
<p align="center">
<img width = "450" src="https://user-images.githubusercontent.com/46090276/206492667-96a1b72a-a8d6-41fa-9b80-07046839fdd5.JPG" alt="Corresponding_Points">
</p>

After warping `Img2.jpg` on `Img1.jpg`, the user should tune $\sigma_{LowPass}, \sigma_{HighPass}, \alpha, \beta$ to achieve a good hybrid image. Hybrid Image is produces by the equation below:
$$HybridImage = \mathscr{F}^{-1}(\alpha H_{LP}[\mathscr{F}(img_1)] + \beta H_{HP}[\mathscr{F}(img_2)])$$

The image below shows the aforementioned parameters that are tuned by user:
<p align="center">
<img width = "450" src="https://user-images.githubusercontent.com/46090276/206498910-13bb96cc-aae9-4e7b-9f30-52adc799f083.JPG" alt="Parameters">
</p>

**Note**: in order to close the pop-up windows, `esc` button should be pressed.

Results
----------

Image1             |  Image 2 |  Result
:-------------------------:|:-------------------------:|:-------------------------:
<img width = "200" src="https://user-images.githubusercontent.com/46090276/206499760-9127ac61-cabc-4ff1-9d67-afafe81d5c27.jpg" alt="res01">  |  <img width = "200" src="https://user-images.githubusercontent.com/46090276/206499783-f48d3e7c-c325-4228-a913-ff2223cfa49f.jpg" alt="res02">   |   <img width = "200" src="https://user-images.githubusercontent.com/46090276/206500629-3f5997ae-1002-498e-8646-d76fe73aeedd.jpg" alt="res04">

Image1             |  Image 2 |  Result
:-------------------------:|:-------------------------:|:-------------------------:
<img width = "200" src="https://user-images.githubusercontent.com/46090276/206499760-9127ac61-cabc-4ff1-9d67-afafe81d5c27.jpg" alt="res01">




References
---------
[1] A. Oliva, A. Torralba, and P. G. Schyns, “Hybrid images,” *ACM SIGGRAPH 2006 Papers*, p. 527–532,
Jul. 2006
