#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <Eigen/Dense>
#include <lodepng.h>
#include <json/json.hpp>

#include "BVHNode.hpp"
#include "Triangle.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "PointLight.hpp"
#include "DirectionalLight.hpp"
#include "LambertianShader.hpp"
#include "TexturedLambertianShader.hpp"
#include "PhongShader.hpp"
#include "MirrorShader.hpp"
#include "TexCoordTestShader.hpp"
#include "Model.hpp"

/// <summary>
/// Load a JSON config file using the nlohmann library.
/// </summary>
nlohmann::json loadConfig(const std::string& filename)
{
	std::ifstream configStream(filename);
	nlohmann::json config = nlohmann::json::parse(configStream);
	return config;
}

/// <summary>
/// Load an Eigen Vector3f from a config file.
/// Call as for example loadVec3FromConfig(config["myVector3"]);
/// </summary>
Eigen::Vector3f loadVec3FromConfig(const nlohmann::json& config)
{
	return Eigen::Vector3f(config[0], config[1], config[2]);
}

int main(int argc, char* argv[]) {

	// *** Load the config file ***
	auto config = loadConfig("../config/config.json");

	const int pixHeight = config["pixHeight"], pixWidth = config["pixWidth"];
	const int nChannels = 4;

	// *** Set up camera and output image ***
	Camera cam(
		loadVec3FromConfig(config["cameraPos"]),
		loadVec3FromConfig(config["cameraForward"]),
		loadVec3FromConfig(config["cameraUp"]),
		pixWidth, pixHeight,
		config["cameraFov"]);

	std::vector<uint8_t> outImage(pixHeight * pixWidth * nChannels);

	Eigen::Vector3f
		red(1.f, 0.f, 0.f),
		blue(0.f, 0.f, 1.f),
		aqua(0.f, .8f, .8f),
		lavender(178.f / 255.f, 164.f / 255.f, 212.f / 255.f);

	// *** Load shaders and textures ***
	std::vector<uint8_t> spotTexture;
	unsigned int width, height;
	lodepng::decode(spotTexture, width, height, "../models/spot.png");

	LambertianShader redLambertianShader(red);
	PhongShader bluePlasticShader(blue, Eigen::Vector3f(1.f, 1.f, 1.f), 100.f);
	LambertianShader aquaLambertianShader(aqua);
	LambertianShader lavenderLambertianShader(lavender);
	TexturedLambertianShader spotShader(&spotTexture, width, height);
	MirrorShader mirrorShader;
	TexCoordTestShader texCoordTestShader;

	// *** Set up scene ***
	Scene scene;

	Model CharacterMesh1("../models/Spidey_Head.obj");
	Model CharacterMesh2("../models/Spidey_Torso.obj");
	Model CharacterMesh3("../models/Spidey_Arms.obj");
	Model CharacterMesh4("../models/Spidey_legs3.obj");
	Model CharacterMesh5("../models/Asgard_Bed.obj");
	Model CharacterMesh6("../models/Bed_Palenquin.obj");
	Model CharacterMesh7("../models/Asgard_Pillar.obj");
	Model CharacterMesh8("../models/Asgard_Pillar.obj");
	Model CharacterMesh9("../models/Asgard_Floor.obj");

	auto makeTranslation = [](float x, float y, float z) {
		Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
		T(0, 3) = x;
		T(1, 3) = y;
		T(2, 3) = z;
		return T;
		};

	auto makeRotateY = [](float angle) {
		Eigen::Matrix4f R = Eigen::Matrix4f::Identity();
		float c = cos(angle);
		float s = sin(angle);

		R(0, 0) = c;  R(0, 2) = s;
		R(2, 0) = -s; R(2, 2) = c;

		return R;
		};

	LambertianShader fallbackShader(red);

	Eigen::Matrix4f baseTransform =
		makeTranslation(-0.2f, -2.0f, 12.f) * makeRotateY(M_PI - 0.2f);

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh1, &fallbackShader, 4, baseTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh2, &fallbackShader, 4, baseTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh3, &fallbackShader, 4, baseTransform));
	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh4, &fallbackShader, 4, baseTransform));

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh5, &fallbackShader, 4,
		makeTranslation(-0.4f, -2.0f, 12.f) * makeRotateY(M_PI - 0.2f)));

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh6, &fallbackShader, 4,
		makeTranslation(-1.0f, -2.4f, 12.f) * makeRotateY(M_PI - 0.2f)));

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh7, &fallbackShader, 4,
		makeTranslation(-0.2f, -2.4f, 15.f) * makeRotateY(M_PI - 0.2f)));

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh8, &fallbackShader, 4,
		makeTranslation(-7.0f, -3.0f, 10.f) * makeRotateY(M_PI - 0.2f)));

	scene.renderables.push_back(std::make_shared<BVHNode>(CharacterMesh9, &fallbackShader, 4,
		makeTranslation(-0.4f, -2.4f, 12.f) * makeRotateY(M_PI - 0.2f)));

	// *** Add lights to scene ***
	Eigen::Vector3f ambientLight(.1f, .1f, .1f);

	std::vector<std::unique_ptr<Light>> lightSources;
	lightSources.push_back(std::make_unique<PointLight>(Eigen::Vector3f(-1.f, 3.f, -1.f), 3.f * Eigen::Vector3f(1.f, 1.f, 1.f)));
	lightSources.push_back(std::make_unique<DirectionalLight>(Eigen::Vector3f(0.f, -1.f, 1.f), .5f * Eigen::Vector3f(1.f, 1.f, 1.f)));

	// *** Render the scene ***

	std::vector<unsigned int> scanlines(pixHeight);
	for (int i = 0; i < pixHeight; ++i) scanlines[i] = i;

	if (config["shuffleScanlines"]) {
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(scanlines.begin(), scanlines.end(), g);
	}

	auto startTime = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int y = 0; y < pixHeight; ++y) {
		for (int x = 0; x < pixWidth; ++x) {

			Ray ray = cam.getRay(x, scanlines[y]);
			HitInfo hitInfo;

			if (scene.intersect(ray, 1e-6f, 1e6f, hitInfo, VISIBLE_BITMASK)) {

				Eigen::Vector3f color = hitInfo.shader->getColor(
					hitInfo, &scene,
					lightSources, ambientLight,
					0, config["maxBounces"]);

				color = color.cwiseMin(1.f);

				int line = (pixHeight - scanlines[y]) - 1;

				outImage[(x + line * pixWidth) * nChannels + 0] = (uint8_t)(color.x() * 255);
				outImage[(x + line * pixWidth) * nChannels + 1] = (uint8_t)(color.y() * 255);
				outImage[(x + line * pixWidth) * nChannels + 2] = (uint8_t)(color.z() * 255);
				outImage[(x + line * pixWidth) * nChannels + 3] = 255;
			}
			else {

				int line = (pixHeight - scanlines[y]) - 1;

				outImage[(x + line * pixWidth) * nChannels + 0] = 0;
				outImage[(x + line * pixWidth) * nChannels + 1] = 0;
				outImage[(x + line * pixWidth) * nChannels + 2] = 0;
				outImage[(x + line * pixWidth) * nChannels + 3] = 255;
			}
		}
	}

	auto renderTime = std::chrono::steady_clock::now() - startTime;

	std::cout << "Render duration " << std::chrono::duration_cast<std::chrono::milliseconds>(renderTime).count() * 1e-3f << " seconds." << std::endl;

	// *** Save the output image ***
	int errorCode;
	errorCode = lodepng::encode(config["outputFilename"], outImage, pixWidth, pixHeight);
	if (errorCode) {
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	return 0;
}