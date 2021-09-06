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
// Angle of repose test:
// step 1: generate a cloud of particles
// step 2: deposit particles into a funnel/slope
// step 3: open the bottom of the funnel
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

// find indices of moving plate particles
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

// update particle position, curr_pos = initial_position + velo * time
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

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <rolling friction coefficient> " +
                     " <sphere-sphere sliding friction coefficient> "
              << std::endl;
}

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = false;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        ShowUsage(argv[0]);
        return 1;
    }

    // CGS UNIT
    double W = 17.7;
    double L = 1.8;
    double H = 12.6;
    double h = 5.6;
    double w = 1.3;
    double theta_star = 50;
    double theta_star_rad = 50.0 / 180.0 * CH_C_PI;

    double rolling_fr_s2s = std::atof(argv[1]);
    double rolling_fr_s2w = 2.f * rolling_fr_s2s;

    double mu_s2s = std::atof(argv[2]);
    double mu_s2w = 1.5 * mu_s2s;

    bool use_material_based_model = true;

    char output_dir[100];
    switch (use_material_based_model) {
        case true:
            sprintf(output_dir, "mur_%.2f_mus_%.2f", rolling_fr_s2s, mu_s2s);
            break;
        case false:
            sprintf(output_dir, "userDefined_mu_%.2f", rolling_fr_s2s);
    }

    std::string out_dir = GetChronoOutputPath() + "repose_wallFric/";
    filesystem::create_directory(filesystem::path(out_dir));
    out_dir = out_dir + output_dir;
    filesystem::create_directory(filesystem::path(out_dir));

    double sphere_radius = 0.05;  // diameter 0.1 cm = 1mm
    double step_size = 1e-6;

    // double sphere_radius = 0.1;  // diameter 0.1 cm = 1mm
    // double step_size = 2e-6;

    // double sphere_radius = L/3;  // diameter 0.1 cm = 1mm
    // double step_size = 1e-5;

    double sphere_density = 2.5;
    double time_setup = 9;
    double time_discharge = 2.5;
    double time_end = time_setup + time_discharge;

    //    double mu_s2w = 0.45;

    double box_X = W;
    double box_Y = 4 * L;
    double box_Z = 2 * H;

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

    // set up bottom and top slope
    std::vector<ChVector<float>> hex_layer;
    float spacing = 2.01 * sphere_radius;
    chrono::utils::HCPSampler<float> sampler(spacing);
    ChVector<float> bottom_center(0.f, 0.f, -H / 2.f + h);
    hex_layer =
        sampler.SampleBox(bottom_center, ChVector<float>(box_X / 2.f - sphere_radius, L / 2.f - sphere_radius, 0.f));

    std::vector<ChVector<float>> bottom_slope;
    std::vector<ChVector<float>> top_slope;
    // moving plate points
    std::vector<ChVector<float>> moving_plate_points;

    double nozzle_width = 0.3;
    for (int i = 0; i < hex_layer.size(); i++) {
        float xp = hex_layer.at(i).x();
        float yp = hex_layer.at(i).y();
        float zp = hex_layer.at(i).z();
        if (xp + sphere_radius < -w / 2.f) {
            double new_z = zp + (-w / 2.f - xp) / std::tan(theta_star_rad);
            bottom_slope.push_back(ChVector<float>(xp, yp, new_z));
            bottom_slope.push_back(ChVector<float>(-xp, yp, new_z));
        }

        else if (xp + 2 * sphere_radius >= -w / 2.f && xp - 2 * sphere_radius <= w / 2.f) {
            moving_plate_points.push_back(ChVector<float>(xp, yp, zp - sphere_radius));
            // std::cout << "nozzle sphere center: " << zp-sphere_radius << std::endl;
        }
    }

    // roughness points on the bottom
    std::vector<ChVector<float>> roughness_points_bottom =
        sampler.SampleBox(ChVector<float>(0, 0, -H / 2.f + sphere_radius),
                          ChVector<float>(box_X / 2.f - sphere_radius, L / 2.f - sphere_radius, 0.f));

    // use hex layer to create top slope (funnel for initialization)
    bottom_center.z() = H / 2.f;
    hex_layer =
        sampler.SampleBox(bottom_center, ChVector<float>(box_X / 2.f - sphere_radius, L / 2.f - sphere_radius, 0.f));

    for (int i = 0; i < hex_layer.size(); i++) {
        float xp = hex_layer.at(i).x();
        float yp = hex_layer.at(i).y();
        float zp = hex_layer.at(i).z();
        if (xp + sphere_radius < -nozzle_width / 2.f) {
            double top_slope_z = zp + (-nozzle_width / 2.f - xp) / std::tan(theta_star_rad);
            if (top_slope_z < H - sphere_radius * 2) {
                top_slope.push_back(ChVector<float>(xp, yp, top_slope_z));
                top_slope.push_back(ChVector<float>(-xp, yp, top_slope_z));
            }
        }
    }

    // generate particle cloud
    std::vector<ChVector<float>> funnel_particle_positions;
    double funnel_z_pos = H / 2.f;
    double particle_z_pos;
    double dim_x;
    std::vector<ChVector<float>> funnel_layer;
    ChVector<float> layer_center(0.0f, 0.0f, 0.0f);
    chrono::utils::PDSampler<float> sampler_pd(2.1 * sphere_radius);

    // add initial particle positions
    while (funnel_z_pos < H - 2 * sphere_radius) {
        funnel_z_pos = funnel_z_pos + 2.1 * sphere_radius;
        layer_center.z() = funnel_z_pos;
        dim_x = std::tan(theta_star_rad) * (funnel_z_pos - H / 2.f) + nozzle_width / 2.f - sphere_radius * 2.f;
        funnel_layer = sampler_pd.SampleBox(layer_center, ChVector<float>(dim_x, L / 2.f - sphere_radius, 0.f));
        funnel_particle_positions.insert(funnel_particle_positions.end(), funnel_layer.begin(), funnel_layer.end());
        std::cout << "created new layer: " << funnel_layer.size() << " particles, total number "
                  << funnel_particle_positions.size() << std::endl;
    }

    // assemble particle position and fixity
    std::vector<ChVector<float>> all_points;
    std::vector<bool> fixed_array;

    fixed_array.insert(fixed_array.end(), bottom_slope.size(), true);
    fixed_array.insert(fixed_array.end(), top_slope.size(), true);
    fixed_array.insert(fixed_array.end(), roughness_points_bottom.size(), true);
    fixed_array.insert(fixed_array.end(), moving_plate_points.size(), true);
    fixed_array.insert(fixed_array.end(), funnel_particle_positions.size(), false);

    all_points.insert(all_points.end(), bottom_slope.begin(), bottom_slope.end());
    all_points.insert(all_points.end(), top_slope.begin(), top_slope.end());
    all_points.insert(all_points.end(), roughness_points_bottom.begin(), roughness_points_bottom.end());
    all_points.insert(all_points.end(), moving_plate_points.begin(), moving_plate_points.end());
    all_points.insert(all_points.end(), funnel_particle_positions.begin(), funnel_particle_positions.end());

    gpu_sys.SetParticles(all_points);
    gpu_sys.SetParticleFixed(fixed_array);
    int NB = all_points.size();
    gpu_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);

    gpu_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gpu_sys.SetRollingCoeff_SPH2SPH(rolling_fr_s2s);
    gpu_sys.SetRollingCoeff_SPH2WALL(rolling_fr_s2w);

    // gpu_sys.SetCohesionRatio(params.cohesion_ratio);
    // gpu_sys.SetAdhesionRatio_SPH2WALL(params.adhesion_ratio_s2w);

    // output initial points
    // just for testing, remove it when done
    char output_filename[100];
    sprintf(output_filename, "funnel_initial_position.csv");
    std::ofstream outstream(std::string(output_filename), std::ios::out);
    outstream << "x, y, z" << std::endl;
    for (int i = 0; i < NB; i++) {
        outstream << all_points.at(i).x() << ", " << all_points.at(i).y() << ", " << all_points.at(i).z() << std::endl;
    }
    outstream.close();

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

    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::EXTENDED_TAYLOR);
    gpu_sys.SetFixedStepSize(step_size);

    gpu_sys.Initialize();

    // find nozzle points and its corresponding initial position in y direction
    std::vector<int> nozzle_point_index;
    std::vector<double> nozzle_point_initial_pos_y;
    findNozzlePointIndices(gpu_sys, -H / 2.f + h - sphere_radius, nozzle_point_index, nozzle_point_initial_pos_y, NB);

    std::cout << "find nozzle indices: " << nozzle_point_index.size() << std::endl;

    int fps_init = 10;
    int fps_discharge = 1000;  // need more frames due to position update
    float curr_time = 0.f;
    int currframe = 0;

    // write an initial frame
    char filename[100];
    sprintf(filename, "%s/step%06d", out_dir.c_str(), currframe);
    gpu_sys.WriteParticleFile(std::string(filename));

    currframe++;

    // initialization
    while (curr_time < time_setup) {
        float frame_step = 1.f / fps_init;
        gpu_sys.AdvanceSimulation(frame_step);

        std::cout << curr_time << std::endl;

        sprintf(filename, "%s/step%06d", out_dir.c_str(), currframe);
        gpu_sys.WriteParticleFile(std::string(filename));

        curr_time += frame_step;
        currframe++;
    }

    int setup_frames = currframe;
    float setup_time = curr_time;
    double nozzle_velo = -9.f;  // nozzle open velocity 90mm/sec
    int steps_per_frame = 10;   // how often to update the position of nozzle per frame
    ChVector<float> velocity;
    ChVector<float> angularVelo;  // add angular velocity

    while (curr_time < time_end) {
        float frame_step = 1.f / (fps_discharge);
        gpu_sys.AdvanceSimulation(frame_step);

        double KE = 0;
        for (int i = 0; i < NB; i++) {
            velocity = gpu_sys.GetParticleVelocity(i);
            KE = KE + velocity.Length2();
        }
        std::cout << curr_time << ", " << KE << std::endl;

        // output position
        if ((currframe - setup_frames) % steps_per_frame == 0) {
            sprintf(filename, "%s/step%06d", out_dir.c_str(),
                    (int)((currframe - setup_frames) / steps_per_frame) + setup_frames);
            gpu_sys.WriteParticleFile(std::string(filename));
        }

        curr_time += frame_step;
        currframe++;

        // update nozzle point position and velocity
        if (nozzle_velo * (curr_time - setup_time) <= L) {
            updateNozzleKinematics(nozzle_velo, curr_time - setup_time, nozzle_point_initial_pos_y, nozzle_point_index,
                                   gpu_sys);
        }
    }
    return 0;
}
