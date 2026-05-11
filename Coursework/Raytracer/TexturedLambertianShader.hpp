#pragma once
#include "Shader.hpp"

/// <summary>
/// Lambertian reflectance shader that samples albedo values from a texture.
/// The texture should be stored as an image file (TGAImage instance).
/// </summary>
class TexturedLambertianShader : public Shader
{
private:
    const std::vector<uint8_t>* albedoTexture_;
    const int texWidth_, texHeight_;
    bool shadowTest_;
public:
    TexturedLambertianShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, bool shadowTest = true)
        :shadowTest_(shadowTest), albedoTexture_(albedoTexture),
        texWidth_(texWidth), texHeight_(texHeight)
    {
    }

    virtual Eigen::Vector3f getColor(const HitInfo& hitInfo,
        const Renderable* scene,
        const std::vector<std::unique_ptr<Light>>& lights,
        const Eigen::Vector3f& ambientLight,
        int currBounceCount,
        const int maxBounces) const
    {
        Eigen::Vector3f albedo;

        if (!albedoTexture_ || albedoTexture_->empty() || texWidth_ <= 0 || texHeight_ <= 0)
            return Eigen::Vector3f(1, 0, 1);

        Eigen::Vector2f tex = hitInfo.texCoords;

        // ===== UV CLAMP =====
        tex.x() = std::max(0.0f, std::min(0.9999f, tex.x()));
        tex.y() = std::max(0.0f, std::min(0.9999f, tex.y()));

        int pixX = static_cast<int>(tex.x() * texWidth_);
        int pixY = static_cast<int>((1.f - tex.y()) * texHeight_);

        pixX = std::max(pixX, 0);
        pixY = std::max(pixY, 0);

        pixX = std::min(pixX, texWidth_ - 1);
        pixY = std::min(pixY, texHeight_ - 1);


        int idx = (pixX + texWidth_ * pixY) * 4;
        if (idx + 2 >= albedoTexture_->size())
            return Eigen::Vector3f(0, 1, 0);

        albedo.x() = static_cast<float>((*albedoTexture_)[idx + 0]) / 255.f;
        albedo.y() = static_cast<float>((*albedoTexture_)[idx + 1]) / 255.f;
        albedo.z() = static_cast<float>((*albedoTexture_)[idx + 2]) / 255.f;

        Eigen::Vector3f color = coefftWiseMul(albedo, ambientLight);

        for (auto& light : lights) {
            if (shadowTest_) {
                if (!light->visibilityCheck(hitInfo.location, scene))
                    continue;
            }
            Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
            float dotProd = std::max(lightVec.dot(hitInfo.normal), 0.f);
            color += dotProd * coefftWiseMul(light->getIntensity(hitInfo.location), albedo);
        }

        return color;
    }
};