// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2021 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// A column of granular material forms a mound, use material parameters
// =============================================================================

#include <iostream>
#include <string>

#include "GpuDemoUtils.hpp"

#include "chrono/core/ChGlobal.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_gpu/ChGpuData.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/utils/ChGpuVisualization.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = true;

int main(int argc, char* argv[]) {
    std::string out_dir = GetChronoOutputPath() + "comparison/";
    filesystem::create_directory(filesystem::path(out_dir));
    out_dir = out_dir + "Repose_new/";
    filesystem::create_directory(filesystem::path(out_dir));

    // Setup simulation
    float sphere_radius = 0.2;
    float sphere_density = 2.5;
    float box_X = 40.0f;
    float box_Y = 40.0f;
    float box_Z = 40.0f;

    // material property
    double cor = 0.9;
    double youngs_modulus = 7e8;  // 700Mpa = 7e8Pa = 7e9 g/(cms^2)
    double poisson_ratio = 0.3;
    double mu_s = 0.3;
    double mu_r = 0.4;

    double step_size = 1e-5;
    double time_end = 2.0f;
    // double time_end = 4.0f;

    ChSystemGpu gpu_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    gpu_sys.UseMaterialBasedModel(true);
    gpu_sys.SetYoungModulus_SPH(youngs_modulus);
    gpu_sys.SetYoungModulus_WALL(youngs_modulus);
    gpu_sys.SetRestitution_SPH(cor);
    gpu_sys.SetRestitution_WALL(cor);
    gpu_sys.SetPoissonRatio_SPH(poisson_ratio);
    gpu_sys.SetPoissonRatio_WALL(poisson_ratio);

    gpu_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s);

    // gpu_sys.SetRollingMode(CHGPU_ROLLING_MODE::NO_RESISTANCE);
    gpu_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gpu_sys.SetRollingCoeff_SPH2SPH(mu_r);
    gpu_sys.SetRollingCoeff_SPH2WALL(mu_r);

    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -980.0f;
    gpu_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));
    gpu_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);

    gpu_sys.SetBDFixed(true);

    // padding in sampler
    float fill_epsilon = 2.02f;
    // padding at top of fill
    ////float drop_height = 0.f;
    float spacing = fill_epsilon * sphere_radius;
    chrono::utils::PDSampler<float> sampler(spacing);

    // Fixed points on the bottom for roughness
    float bottom_z = -box_Z / 2.f + sphere_radius;
    ChVector<> bottom_center(0, 0, bottom_z);
    std::vector<ChVector<float>> roughness_points = sampler.SampleBox(
        bottom_center, ChVector<float>(box_X / 2.f - sphere_radius, box_Y / 2.f - sphere_radius, 0.f));

    // Create column of material
    std::vector<ChVector<float>> material_points;

    float fill_bottom = bottom_z + spacing;
    float fill_width = 5.f;
    float fill_height = 2.f * fill_width;
    ////float fill_top = fill_bottom + fill_height;

    ChVector<float> center(0.f, 0.f, fill_bottom + fill_height / 2.f);
    material_points = sampler.SampleCylinderZ(center, fill_width, fill_height / 2.f);

    std::vector<ChVector<float>> body_points;
    std::vector<bool> body_points_fixed;
    body_points.insert(body_points.end(), roughness_points.begin(), roughness_points.end());
    body_points_fixed.insert(body_points_fixed.end(), roughness_points.size(), true);

    body_points.insert(body_points.end(), material_points.begin(), material_points.end());
    body_points_fixed.insert(body_points_fixed.end(), material_points.size(), false);

    gpu_sys.SetParticles(body_points);
    gpu_sys.SetParticleFixed(body_points_fixed);

    std::cout << "Added " << roughness_points.size() << " fixed points" << std::endl;
    std::cout << "Added " << material_points.size() << " material points" << std::endl;

    std::cout << "Actually added " << body_points.size() << std::endl;

    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::EXTENDED_TAYLOR);
    gpu_sys.SetFixedStepSize(step_size);

    gpu_sys.Initialize();

    int fps = 100;
    float frame_step = 1.f / fps;
    float curr_time = 0.f;
    int currframe = 0;

    // write an initial frame char filename[100];
    char particle_filename[100];
    // sprintf(particle_filename, "%s/step%06d.csv", out_dir.c_str(), currframe);
    // gpu_sys.WriteParticleFile(std::string(particle_filename));
    // currframe++;

    FILE* pFile;
    char filename[100];
    int stupid_ball_id = 3588;
    sprintf(filename, "gpu_id_%d.csv", stupid_ball_id);
    pFile = fopen(filename, "w");

    ChVector<float> ball_velo;
    ChVector<float> ball_angular_velo;

    while (curr_time < time_end) {
        gpu_sys.AdvanceSimulation(frame_step);
        std::cout << curr_time << std::endl;
        sprintf(particle_filename, "%s/step%06d.csv", out_dir.c_str(), currframe);
        gpu_sys.WriteParticleFile(std::string(particle_filename));

        // ball_velo = gpu_sys.GetParticleVelocity(stupid_ball_id);
        // ball_angular_velo = gpu_sys.GetParticleAngVelocity(stupid_ball_id);
        //
        // fprintf(pFile, "%e, %e, %e, %e, %e, %e, %e\n", curr_time, ball_velo.x(), ball_velo.y(), ball_velo.z(),
        //         ball_angular_velo.x(), ball_angular_velo.y(), ball_angular_velo.z());
        curr_time += frame_step;
        currframe++;
    }

    return 0;
}
