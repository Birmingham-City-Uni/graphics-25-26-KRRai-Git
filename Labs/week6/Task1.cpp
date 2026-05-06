// This define is necessary to get the M_PI constant.
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <lodepng.h>
#include "Image.hpp"
#include "LinAlg.hpp"
#include "Light.hpp"
#include "Mesh.hpp"

// ***** WEEK 6 LAB *****
// Subtask 1: Implement the projectionMatrix function, to make a projection matrix to view your scene!
// Subtask 2: Complete the transformation chain, moving vertices from model space all the way to screen space.
// Subtask 3: Set up the camera and projection matrices for the transformation chain
// Subtask 4: Implement Z buffering.
// Subtask 5: Implement texture mapping.
// If you finish early - note that we now have all the tools to properly set up your own scene!
// This is a great time to start on your own code in the coursework/rasteriser folder, using this as a base if
// you wish. We will in future labs work on more advanced shading, but you can port this feature over later.


struct Triangle {
	std::array<Eigen::Vector3f, 3> screen; // Coordinates of the triangle in screen space.
	std::array<Eigen::Vector3f, 3> verts; // Vertices of the triangle in world space.
	std::array<Eigen::Vector3f, 3> norms; // Normals of the triangle corners in world space.
	std::array<Eigen::Vector2f, 3> texs; // Texture coordinates of the triangle corners.
};


Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 70.f * M_PI / 180.f, float zFar = 10.f, float zNear = 0.1f)
{
	// ========= Subtask 1: Make a Projection Matrix ========
	// *** YOUR CODE HERE ***

	float aspect = (float)width / (float)height;
	float vertFov = 2.0f * atan(tan(horzFov / 2.0f) / aspect);

	float f = 1.0f / tan(vertFov / 2.0f);

	Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(2, 3) = (2 * zFar * zNear) / (zNear - zFar);
	projection(3, 2) = -1.0f;

	return projection;
	// *** END YOUR CODE ***
}

void findScreenBoundingBox(const Triangle& t, int width, int height, int& minX, int& minY, int& maxX, int& maxY)
{
	// Find a bounding box around the triangle
	minX = std::min(std::min(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	minY = std::min(std::min(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());
	maxX = std::max(std::max(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	maxY = std::max(std::max(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());

	// Constrain it to lie within the image.
	minX = std::min(std::max(minX, 0), width - 1);
	maxX = std::min(std::max(maxX, 0), width - 1);
	minY = std::min(std::max(minY, 0), height - 1);
	maxY = std::min(std::max(maxY, 0), height - 1);
}


void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer,
	const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) {
		// Triangle is backfacing
		// Exit and quit drawing!
		return;
	}

	for (int x = minX; x <= maxX; ++x)
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			float sum = b0 + b1 + b2;
			if (sum > 1.0001) {
				continue;
			}

			Eigen::Vector3f worldP = t.verts[0] * b0 + t.verts[1] * b1 + t.verts[2] * b2;

			float depth = t.screen[0].z() * b0 + t.screen[1].z() * b1 + t.screen[2].z() * b2;

			int depthIdx = y * width + x;

			if (depth > zBuffer[depthIdx]) {
				continue;
			}
			zBuffer[depthIdx] = depth;

			Eigen::Vector3f normP = t.norms[0] * b0 + t.norms[1] * b1 + t.norms[2] * b2;
			normP.normalize();

			Eigen::Vector2f texP = t.texs[0] * b0 + t.texs[1] * b1 + t.texs[2] * b2;

			int texC = (int)(texP.x() * texWidth);
			int texR = (int)((1.0f - texP.y()) * texHeight);

			texC = std::max(0, std::min(texC, texWidth - 1));
			texR = std::max(0, std::min(texR, texHeight - 1));

			Color texColor = getPixel(albedoTexture, texC, texR, texWidth, texHeight);

			Eigen::Vector3f albedo;
			albedo << texColor.r / 255.0f,
				texColor.g / 255.0f,
				texColor.b / 255.0f;
			albedo = albedo.array().pow(2.2f);

			Eigen::Vector3f color = Eigen::Vector3f::Zero();

			for (auto& light : lights) {

				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);

				if (light->getType() != Light::Type::AMBIENT) {

					float dotProd = normP.dot(-light->getDirection(worldP));

					dotProd = std::max(dotProd, 0.0f);

					lightIntensity *= dotProd;
				}

				color += coeffWiseMultiply(lightIntensity, albedo);
			}

			Color c;
			c.r = std::min(powf(color.x(), 1 / 2.2f), 1.0f) * 255;
			c.g = std::min(powf(color.y(), 1 / 2.2f), 1.0f) * 255;
			c.b = std::min(powf(color.z(), 1 / 2.2f), 1.0f) * 255;

			c.a = 255;

			setPixel(image, x, y, width, height, c);
		}
}



void drawMesh(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height)
{
	for (int i = 0; i < mesh.vFaces.size(); ++i) {

		Eigen::Vector3f
			v0 = mesh.verts[mesh.vFaces[i][0]],
			v1 = mesh.verts[mesh.vFaces[i][1]],
			v2 = mesh.verts[mesh.vFaces[i][2]];
		Eigen::Vector3f
			n0 = mesh.norms[mesh.nFaces[i][0]],
			n1 = mesh.norms[mesh.nFaces[i][1]],
			n2 = mesh.norms[mesh.nFaces[i][2]];

		Triangle t;
		t.verts[0] = (modelToWorld * vec3ToVec4(v0)).block<3, 1>(0, 0);
		t.verts[1] = (modelToWorld * vec3ToVec4(v1)).block<3, 1>(0, 0);
		t.verts[2] = (modelToWorld * vec3ToVec4(v2)).block<3, 1>(0, 0);

		Eigen::Vector4f vClip0 = worldToClip * vec3ToVec4(t.verts[0]);
		Eigen::Vector4f vClip1 = worldToClip * vec3ToVec4(t.verts[1]);
		Eigen::Vector4f vClip2 = worldToClip * vec3ToVec4(t.verts[2]);

		vClip0 /= vClip0.w();
		vClip1 /= vClip1.w();
		vClip2 /= vClip2.w();

		if (outsideClipBox(vClip0) && outsideClipBox(vClip1) && outsideClipBox(vClip2)) {
			continue;
		}

		auto toScreen = [&](Eigen::Vector4f v) {
			float x = (v.x() + 1.0f) * 0.5f * width;
			float y = (1.0f - (v.y() + 1.0f) * 0.5f) * height;
			return Eigen::Vector3f(x, y, v.z());
			};

		t.screen[0] = toScreen(vClip0);
		t.screen[1] = toScreen(vClip1);
		t.screen[2] = toScreen(vClip2);

		t.norms[0] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n0).normalized();
		t.norms[1] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n1).normalized();
		t.norms[2] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * n2).normalized();

		t.texs[0] = mesh.texs[mesh.tFaces[i][0]];
		t.texs[1] = mesh.texs[mesh.tFaces[i][1]];
		t.texs[2] = mesh.texs[mesh.tFaces[i][2]];

		drawTriangle(image, width, height, zBuffer, t, lights, albedoTexture, texWidth, texHeight);
	}
}


int main()
{
	std::string outputFilename = "output.png";

	const int width = 512, height = 512;
	const int nChannels = 4;

	std::vector<uint8_t> imageBuffer(height * width * nChannels);
	std::vector<float> zBuffer(height * width);

	Color black{ 0,0,0,255 };
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			setPixel(imageBuffer, c, r, width, height, black);
			zBuffer[r * width + c] = 1.0f;
		}
	}

	Eigen::Matrix4f projection = projectionMatrix(height, width);

	Eigen::Matrix4f cameraToWorld = translationMatrix(Eigen::Vector3f(0.f, 0.8f, 0.0f)) * rotateXMatrix(0.4f) * rotateYMatrix(M_PI);

	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
	Eigen::Matrix4f worldToClip = projection * worldToCamera;

	std::string bunnyFilename = "../models/stanford_bunny_texmapped.obj";

	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.1f, 0.1f, 0.1f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(0.4f, 0.4f, 0.4f), Eigen::Vector3f(1.f, 0.f, 0.0f)));

	Mesh bunnyMesh = loadMeshFile(bunnyFilename);

	Eigen::Matrix4f bunnyTransform;

	std::vector<uint8_t> bunnyTexture;
	unsigned int bunnyTexWidth, bunnyTexHeight;
	lodepng::decode(bunnyTexture, bunnyTexWidth, bunnyTexHeight, "../models/stanford_bunny_albedo.png");

	bunnyTransform = translationMatrix(Eigen::Vector3f(-1.0f, -1.0f, 3.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);
	bunnyTransform = translationMatrix(Eigen::Vector3f(-1.0f, -1.0f, 5.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);
	bunnyTransform = translationMatrix(Eigen::Vector3f(-1.0f, -1.0f, 7.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);
	bunnyTransform = translationMatrix(Eigen::Vector3f(1.0f, -1.0f, 3.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);
	bunnyTransform = translationMatrix(Eigen::Vector3f(1.0f, -1.0f, 5.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);
	bunnyTransform = translationMatrix(Eigen::Vector3f(1.0f, -1.0f, 7.f)) * rotateYMatrix(0.0f) * rotateXMatrix(0.8f);
	drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight, bunnyTransform, worldToClip, lights, width, height);

	drawPointLights(imageBuffer, width, height, lights);

	int errorCode;
	errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) {
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	saveZBufferImage("zBuffer.png", zBuffer, width, height);

	return 0;
}