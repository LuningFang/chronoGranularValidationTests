// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// piling test of granular material from Cecily et al 2020
// =============================================================================

// todo
// find index
// assign velocity and position
// position use offset!!!
// need to record initial position of those particles
// add front and back plate

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
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

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <rolling friction coefficient> " + " <use material-based property> " << std::endl;
}

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = false;

void findNozzlePointIndices(ChSystemGpu& gpu_sys,
                            double bottom_z,
                            std::vector<int>& nozzle_point_index,
                            std::vector<double>& nozzle_point_initial_pos_y,
                            int nb) {
    for (int i = 0; i < nb; i++) {
        ChVector<float> pos = gpu_sys.GetParticlePosition(i);
        double z = pos.z();
        if (std::abs(z - bottom_z) < 1e-5) {
            nozzle_point_initial_pos_y.push_back(pos.y());
            nozzle_point_index.push_back(i);
        }
    }
}

void updateNozzleKinematics(double velo,
                            double time,
                            std::vector<double> initial_position,
                            std::vector<int> nozzle_index,
                            ChSystemGpu& gpu_sys) {
    int index;
    ChVector<double> position;
    ChVector<double> velocity;

    for (int i = 0; i < nozzle_index.size(); i++) {
        index = nozzle_index.at(i);
        position = gpu_sys.GetParticlePosition(index);
        velocity = gpu_sys.GetParticleVelocity(index);
        position.y() = initial_position.at(i) + velo * time;
        velocity.y() = velo;
        gpu_sys.SetParticleVelocity(index, velocity);
        gpu_sys.SetParticlePosition(index, position);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        ShowUsage(argv[0]);
        return 1;
    }

    bool use_material_based_model;

    if (std::atoi(argv[2]) == 0) {
        use_material_based_model = false;
    } else {
        use_material_based_model = true;
    }

    double rolling_fr_s2s = std::atof(argv[1]);
    double rolling_fr_s2w = rolling_fr_s2s;

    // CGS UNIT
    double W = 17.7;
    double L = 1.8;
    double H = 12.6;
    double h = 5.6;
    double w = 1.3;
    double theta_star = 50;
    double theta_star_rad = 50.0 / 180.0 * CH_C_PI;

    char output_dir[100];
    switch (use_material_based_model) {
        case true:
            sprintf(output_dir, "matBased_mu_%.2f", rolling_fr_s2s);
            break;
        case false:
            sprintf(output_dir, "userDefined_mu_%.2f", rolling_fr_s2s);
    }

    std::string out_dir = GetChronoOutputPath() + "repose_oldway/";
    std::cout << "output directory: " << out_dir << std::endl;
    filesystem::create_directory(filesystem::path(out_dir));
    out_dir = out_dir + output_dir;
    filesystem::create_directory(filesystem::path(out_dir));

    double sphere_radius = 0.05;  // diameter 0.1 cm = 1mm
    double step_size = 1e-6;

    // double sphere_radius = 0.1;  // diameter 0.1 cm = 1mm
    // double step_size = 5e-6;

    double sphere_density = 2.5;
    double time_end = 4.5;

    double mu_s2s = 0.16;
    double mu_s2w = 0.45;

    double box_X = W;
    double box_Y = 4 * L;
    double box_Z = H * 1.2;

    double grav_X = 0.0;
    double grav_Y = 0.0;
    double grav_Z = -981;

    // Setup simulation
    ChSystemGpu gpu_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    switch (use_material_based_model) {
        case true: {
            double cor_p = 0.97;
            double cor_w = 0.82;
            double youngs_modulus = 7e8;  // 70Mpa = 7e7Pa = 7e8 g/(cms^2)
            double poisson_ratio = 0.24;

            gpu_sys.UseMaterialBasedModel(true);

            gpu_sys.SetYoungModulus_SPH(youngs_modulus);
            gpu_sys.SetYoungModulus_WALL(youngs_modulus);

            gpu_sys.SetRestitution_SPH(cor_p);
            gpu_sys.SetRestitution_WALL(cor_w);

            gpu_sys.SetPoissonRatio_SPH(poisson_ratio);
            gpu_sys.SetPoissonRatio_WALL(poisson_ratio);

            break;
        }

        case false: {
            double kn = 1e7;
            double gn = 2e4;
            double kt = 2e6;
            double gt = 50;

            gpu_sys.SetKn_SPH2SPH(kn);
            gpu_sys.SetKn_SPH2WALL(kn);
            gpu_sys.SetGn_SPH2SPH(gn);
            gpu_sys.SetGn_SPH2WALL(gn);

            gpu_sys.SetKt_SPH2SPH(kt);
            gpu_sys.SetKt_SPH2WALL(kt);
            gpu_sys.SetGt_SPH2SPH(gt);
            gpu_sys.SetGt_SPH2WALL(gt);

            gpu_sys.UseMaterialBasedModel(false);
        }
    }

    gpu_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);

    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);

    gpu_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gpu_sys.SetRollingCoeff_SPH2SPH(rolling_fr_s2s);
    gpu_sys.SetRollingCoeff_SPH2WALL(rolling_fr_s2w);

    gpu_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));
    gpu_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    gpu_sys.SetBDFixed(true);

    // front and back plane at location (0, +/- L/2.0f, 0)
    ChVector<float> front_plate_pos(0.0, L / 2.0f, 0.0);
    ChVector<float> front_plate_normal(0.0, -1.0f, 0.0f);
    size_t front_plate_id = gpu_sys.CreateBCPlane(front_plate_pos, front_plate_normal, true);

    ChVector<float> back_plate_pos(0.0, -L / 2.0f, 0.0);
    ChVector<float> back_plate_normal(0.0, 1.0f, 0.0f);
    size_t back_plate_id = gpu_sys.CreateBCPlane(back_plate_pos, back_plate_normal, true);

    // Add roughness points here
    float bottom_z = -H / 2.f + sphere_radius;
    ChVector<> bottom_center(0, 0, bottom_z);

    // set up funnel as numerical boundary with roughness points
    std::vector<ChVector<float>> hex_layer;
    float fill_epsilon = 2.01f;
    float spacing = fill_epsilon * sphere_radius;
    chrono::utils::HCPSampler<float> sampler(spacing);

    ChVector<float> nozzle_center(0.f, 0.f, -H / 2.f + h);
    hex_layer =
        sampler.SampleBox(nozzle_center, ChVector<float>(box_X / 2.f - sphere_radius, L / 2.f - sphere_radius, 0.f));

    // roughness points
    std::vector<ChVector<float>> roughness_points;
    // moving plate points
    std::vector<ChVector<float>> moving_plate_points;
    for (int i = 0; i < hex_layer.size(); i++) {
        float xp = hex_layer.at(i).x();
        float yp = hex_layer.at(i).y();
        float zp = hex_layer.at(i).z();
        if (xp + sphere_radius < -w / 2.f) {
            zp += (-w / 2.f - xp) / std::tan(theta_star_rad);
            roughness_points.push_back(ChVector<float>(xp, yp, zp));
            roughness_points.push_back(ChVector<float>(-xp, yp, zp));
        }

        else if (xp + sphere_radius >= -w / 2.f && xp + sphere_radius < 0.0f) {
            moving_plate_points.push_back(ChVector<float>(xp, yp, zp - sphere_radius));
            moving_plate_points.push_back(ChVector<float>(-xp, yp, zp - sphere_radius));
        }
    }

    // add nozzle plate points to roughness array
    roughness_points.insert(roughness_points.end(), moving_plate_points.begin(), moving_plate_points.end());

    // roughness points on the bottom
    std::vector<ChVector<float>> roughness_points_bottom =
        sampler.SampleBox(bottom_center, ChVector<float>(box_X / 2.f - sphere_radius, L / 2.f - sphere_radius, 0.f));
    roughness_points.insert(roughness_points.end(), roughness_points_bottom.begin(), roughness_points_bottom.end());

    // load input initial position
    std::vector<ChVector<float>> material_points;
    string inputPosFileName = GetChronoDataFile("models/repose/initial_position_49k.csv");
    material_points = loadPositionCheckpoint<float>(inputPosFileName);
    std::cout << "number of material points " << material_points.size() << std::endl;

    std::vector<ChVector<float>> body_points;
    std::vector<bool> fixed_array;

    body_points.insert(body_points.end(), roughness_points.begin(), roughness_points.end());
    body_points.insert(body_points.end(), material_points.begin(), material_points.end());

    fixed_array.insert(fixed_array.end(), roughness_points.size(), true);
    fixed_array.insert(fixed_array.end(), material_points.size(), false);

    gpu_sys.SetParticles(body_points);
    gpu_sys.SetParticleFixed(fixed_array);

    std::cout << "Added " << body_points.size() << " particles" << std::endl;

    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CHUNG);
    gpu_sys.SetFixedStepSize(step_size);

    gpu_sys.Initialize();

    // find nozzle points and its corresponding initial position in y direction
    std::vector<int> nozzle_point_index;
    std::vector<double> nozzle_point_initial_pos_y;
    findNozzlePointIndices(gpu_sys, nozzle_center.z() - sphere_radius, nozzle_point_index, nozzle_point_initial_pos_y,
                           body_points.size());

    int fps = 1000;
    float frame_step = 1.f / fps;
    float curr_time = 0.f;
    int currframe = 0;
    unsigned int total_frames = (unsigned int)((float)time_end * fps);

    // write an initial frame
    char filename[100];
    sprintf(filename, "%s/step%06d", out_dir.c_str(), currframe);
    gpu_sys.WriteParticleFile(std::string(filename));

    currframe++;

    std::cout << "frame step is " << frame_step << std::endl;
    ChVector<float> velocity;
    ChVector<float> angularVelo;

    // output time vs force
    // char output_filename[100];
    // sprintf(output_filename, "user_defined_%d_mu_r_%.2f.csv", atoi(argv[2]), atof(argv[1]));
    // std::ofstream outstream(std::string(output_filename), std::ios::out);

    double nozzle_velo = -9.f;  // nozzle open velocity 90mm/sec
    int steps_per_frame = 10;
    while (curr_time < time_end) {
        gpu_sys.AdvanceSimulation(frame_step);

        double KE = 0;
        for (int i = 0; i < body_points.size(); i++) {
            velocity = gpu_sys.GetParticleVelocity(i);
            KE = KE + velocity.Length2();
        }
        std::cout << curr_time << ", " << KE << std::endl;

        // ChVector<float> plane_reaction_force;

        // gpu_sys.GetBCReactionForces(bottom_plate_id, plane_reaction_force);
        // std::cout << ", bottom plate force: " << plane_reaction_force.x() << ", " << plane_reaction_force.y() << ", "
        // << plane_reaction_force.z() << std::endl;

        if (currframe % steps_per_frame == 0) {
            sprintf(filename, "%s/step%06d", out_dir.c_str(), (int)(currframe / steps_per_frame) + 1);
            gpu_sys.WriteParticleFile(std::string(filename));
        }

        // std::cout << std::setprecision(7) << curr_time << ", " << plane_reaction_force.z() << std::endl;
        curr_time += frame_step;
        currframe++;

        if (nozzle_velo * (curr_time - 2) <= L && curr_time > 2.0f) {
            updateNozzleKinematics(nozzle_velo, curr_time - 2, nozzle_point_initial_pos_y, nozzle_point_index, gpu_sys);
        }
    }

    // outstream.close();
    return 0;
}
