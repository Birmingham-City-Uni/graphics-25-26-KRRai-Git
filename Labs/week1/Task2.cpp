#include <iostream>
#include <lodepng.h>


int main()
{
	std::string inputFilename = "./images/stanford_bunny.png";
	std::string outputFilename = "output_bunny.png";

	std::vector<uint8_t> imageBuffer;
	unsigned int width, height;
	lodepng::decode(imageBuffer, width , height , inputFilename);


	width = width / 2;
	height = height / 2;


	// *** Tasks ***
	// This code loads an image from a png file. This is an image of the famous 
	// Stanford Bunny https://engineering.stanford.edu/magazine/tale-ubiquitous-stanford-bunny
	// You'll need to load and manipulate images to add texturing to your rasteriser and raytracer.
	// Let's try changing this image.
	// If you'd like, you can use the setPixel function you wrote in the previous task.
	// The code below reduces the brightness of the image by 0.5x, as an example.


	for(int y = 0; y < height; ++y) 
		for (int x = 0; x < width; ++x) 
			for (int c = 0; c < 3; ++c) {
				int pixelIdx = x + y * width;
				imageBuffer[pixelIdx * 4 + c] = 255 - imageBuffer[pixelIdx * 4 + c];
			}

	// Once you have tested this code, comment out the for loops above and try the following tasks:
	// * Task 1: Try making a *negative* of the input image. Pixels that are bright in the input
	//           should be dark in your output. 
	//           Hint: if the pixels ranged in value from 0 to 1, you could replace each pixel value (v) with (1 - v). 
	//           but remember, the pixels have 8-bit unsigned values, so range from 0 to 255!
	// * Optional Task 2: Try downsampling this image to 1/2 resolution.
	//           You can either just keep one in every 4 pixels, or better yet, average the pixels in each 2x2 block.
	//           Hint: Be careful when averaging! You probably want to convert the pixel values to floating-point to
	//           do the averaging maths.

	int errorCode;
	errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) { // check the error code, in case an error occurred.
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	return 0;
}
