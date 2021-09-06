// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// used to initialize things
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>
#include <iomanip>
#include <ctime>

#include "chrono/core/ChGlobal.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/core/ChMathematics.h"

#include "chrono_gpu/ChGpuData.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/utils/ChGpuVisualization.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "GpuDemoUtils.hpp"

using namespace chrono;
using namespace chrono::gpu;

std::vector<ChVector<float>> generateBoundaryLayer(double cylinder_radius,
                                                   double sphere_diameter,
                                                   double cylinder_width) {
    int numLayers = (int)(cylinder_width / sphere_diameter);
    int particles_per_layer = (int)(std::ceil(2 * CH_C_PI * cylinder_radius / sphere_diameter));

    double theta_interval = 2 * CH_C_PI / (double)particles_per_layer;
    std::vector<ChVector<float>> boundary_layer;

    double particle_x, particle_y, particle_z;
    double theta;
    for (int iz = 0; iz < numLayers; iz++) {
        particle_z = -cylinder_width / 2.0f + sphere_diameter * 0.5f + iz * sphere_diameter;

        for (int j = 0; j < particles_per_layer; j++) {
            theta = j * theta_interval;
            particle_x = cos(theta) * (cylinder_radius - sphere_diameter / 2.0f);
            particle_y = sin(theta) * (cylinder_radius - sphere_diameter / 2.0f);

            boundary_layer.push_back(ChVector<float>(particle_x, particle_y, particle_z));
        }
    }
    std::cout << "Created " << boundary_layer.size() << "particles for boundary" << std::endl;
    return boundary_layer;
}

// boundary layer position update
void updateBoundaryLayerPosition(double cyl_radius,
                                 std::vector<int> boundary_idx,
                                 double omega,
                                 ChSystemGpu& gpu_sys,
                                 double dt) {
    int index;
    ChVector<float> position;
    double vx, vy;
    double sin_theta, cos_theta;
    ChVector<double> position_curr;
    for (int i = 0; i < boundary_idx.size(); i++) {
        index = boundary_idx.at(i);
        position = gpu_sys.GetParticlePosition(index);
        sin_theta = position.y() / cyl_radius;
        cos_theta = position.x() / cyl_radius;
        vx = -omega * cyl_radius * sin_theta;
        vy = omega * cyl_radius * cos_theta;

        position_curr.x() = position.x() + dt * vx;
        position_curr.y() = position.y() + dt * vy;
        position_curr.z() = position.z();

        gpu_sys.SetParticlePosition(index, position_curr);
    }
}

// boundary layer position update
void updateBoundaryLayerVelocity(double cyl_radius, std::vector<int> boundary_idx, double omega, ChSystemGpu& gpu_sys) {
    int index;
    ChVector<float> position;
    ChVector<float> velocity;
    double sin_theta, cos_theta;
    double vx, vy;
    for (int i = 0; i < boundary_idx.size(); i++) {
        index = boundary_idx.at(i);
        position = gpu_sys.GetParticlePosition(index);

        sin_theta = position.y() / cyl_radius;
        cos_theta = position.x() / cyl_radius;

        vx = -omega * cyl_radius * sin_theta;
        vy = omega * cyl_radius * cos_theta;

        velocity.x() = -omega * cyl_radius * sin_theta;
        velocity.y() = omega * cyl_radius * cos_theta;
        velocity.z() = 0.0f;

        gpu_sys.SetParticleVelocity(index, velocity);
    }
}

// find boundary layer index
std::vector<int> findBoundaryLayerIndex(ChSystemGpu& gpu_sys, double cyl_radius, int nb) {
    std::vector<int> boundary_idx;
    for (int i = 0; i < nb; i++) {
        ChVector<float> pos = gpu_sys.GetParticlePosition(i);
        double dist = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
        if (std::abs(dist - cyl_radius) < 1e-5) {
            boundary_idx.push_back(i);
        }
    }
    return boundary_idx;
}

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <drum rotation speed (rpm)> " << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        ShowUsage(argv[0]);
        return 1;
    }

    // unit cgs
    double drum_diameter = 6;
    double drum_height = 0.5;

    // caltech test
    // double drum_diameter = 30;
    // double drum_height = 1.6;

    // double sphere_radius = 0.05;
    double sphere_radius = 0.0265;
    double sphere_density = 2.5;

    double box_X = drum_diameter;
    double box_Y = drum_diameter;
    double box_Z = drum_height * 1.2;

    // double mu_s2s = 0.16;
    // double mu_s2w = 0.45;

    double mu_s2s = 0.0f;
    double mu_s2w = 0.0f;

    double rolling_fr_s2s = 0.09;
    double rolling_fr_s2w = 0.09;

    double drum_omega_rpm = std::atof(argv[1]);
    //    double drum_omega_rpm = 5.4f;
    double drum_omega = drum_omega_rpm * 2.0f * CH_C_PI / 60.0f;  // drumming spinning rad/s
    // double drum_omega = 55.0f * 2.0f * CH_C_PI / 60.0f; // drumming spinning rad/s
    // double drum_omega = 212.0f * 2.0f * CH_C_PI / 60.0f; // drumming spinning rad/s

    // double drum_omega_deg = 2.0f;
    // double drum_omega = drum_omega_deg / 180.f * CH_C_PI; // drumming spinning rad/s

    double step_size = 1e-6;
    double time_end = 0.5f;
    ChSystemGpu gpu_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    double grav_X = 0.0;
    double grav_Y = -981.0;
    double grav_Z = 0.0;

    bool use_material_based_model = true;
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

    // gpu_sys.SetCohesionRatio(params.cohesion_ratio);
    // gpu_sys.SetAdhesionRatio_SPH2MESH(params.adhesion_ratio_s2m);
    // gpu_sys.SetAdhesionRatio_SPH2WALL(params.adhesion_ratio_s2w);

    gpu_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gpu_sys.SetFrictionMode(chrono::gpu::CHGPU_FRICTION_MODE::MULTI_STEP);

    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);

    gpu_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gpu_sys.SetRollingCoeff_SPH2SPH(rolling_fr_s2s);
    gpu_sys.SetRollingCoeff_SPH2WALL(rolling_fr_s2w);

    std::string out_dir = GetChronoOutputPath() + "tumbler_newrun/";
    filesystem::create_directory(filesystem::path(out_dir));

    char output_dir[100];
    sprintf(output_dir, "spinning_omega_%d_rpm", (int)drum_omega_rpm);

    out_dir = out_dir + output_dir;
    filesystem::create_directory(filesystem::path(out_dir));

    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gpu_sys.SetFixedStepSize(step_size);
    gpu_sys.SetBDFixed(true);
    gpu_sys.SetVerbosity(CHGPU_VERBOSITY::INFO);

    // create front plate at location drum_height/2
    ChVector<float> front_plate_pos(0.0, 0.0, drum_height / 2.0);
    ChVector<float> front_plate_normal(0.0, 0.0, -1.0f);
    size_t front_plate_id = gpu_sys.CreateBCPlane(front_plate_pos, front_plate_normal, true);

    // bottom plate at location -H/2, for recording force
    ChVector<float> back_plate_pos(0.0, 0.0, -drum_height / 2.0);
    ChVector<float> back_plate_normal(0.0, 0.0, 1.0f);
    size_t back_plate_id = gpu_sys.CreateBCPlane(back_plate_pos, back_plate_normal, true);

    ChVector<double> spin_omega(0.0f, 0.0f, drum_omega);
    gpu_sys.SetBCPlaneRotation(front_plate_id, ChVector<double>(0.0f, 0.0f, 0.0f), spin_omega);
    gpu_sys.SetBCPlaneRotation(back_plate_id, ChVector<double>(0.0f, 0.0f, 0.0f), spin_omega);

    // set rotation

    // set cylinder boundary
    // ChVector<float> cyl_center(0, 0, 0);
    // const float cyl_rad = drum_diameter / 2.f;

    // size_t cylinder_id = gpu_sys.CreateBCCylinderZ(cyl_center, cyl_rad, false, true);

    // sampling particles
    // utils::HCPSampler<float> sampler(2.1f * sphere_radius);
    // fill out particles
    // std::vector<ChVector<float>> filler_points;
    // const float filler_bottom = -box_Z/2.0 + sphere_radius;
    // const float filler_radius = box_X / 2.f - sphere_radius*2.0f;
    // const float filler_top = box_Z/2.0 - sphere_radius;

    // ChVector<float> filler_center(0.0, 0.0, filler_bottom);
    // while (filler_center.z() < filler_top) {
    //     auto points = sampler.SampleCylinderZ(filler_center, filler_radius, 0);
    //     filler_points.insert(filler_points.end(), points.begin(), points.end());
    //     filler_center.z() += 2.1f * sphere_radius;
    // }

    std::vector<ChVector<float>> filler_points;

    //    string initial_points_filename =
    //    GetChronoDataFile("models/tumbler_caltech/tumbler_initial_positions_d_1mm_large_stepsize.csv");
    string initial_points_filename = GetChronoDataFile("models/tumbler/tumbler_initial_positions_0.5mm.csv");

    filler_points = loadPositionCheckpoint<float>(initial_points_filename);
    std::cout << "loaded " << filler_points.size() << " particles for filling" << std::endl;

    std::vector<ChVector<float>> boundary_points =
        generateBoundaryLayer(drum_diameter / 2.0f, sphere_radius * 2.0f, drum_height);

    std::vector<bool> fixed_points;
    std::vector<ChVector<float>> all_points;
    all_points.insert(all_points.end(), boundary_points.begin(), boundary_points.end());
    all_points.insert(all_points.end(), filler_points.begin(), filler_points.end());

    fixed_points.insert(fixed_points.end(), boundary_points.size(), true);
    fixed_points.insert(fixed_points.end(), filler_points.size(), false);

    gpu_sys.SetParticles(all_points);
    gpu_sys.SetParticleFixed(fixed_points);

    gpu_sys.Initialize();

    std::vector<int> boundary_idx =
        findBoundaryLayerIndex(gpu_sys, drum_diameter / 2.0f - sphere_radius, all_points.size());

    float curr_time = 0.0f;
    int currframe = 0;

    char filename[100];
    ChVector<float> velocity;

    // write initial position
    sprintf(filename, "%s/step%06d", out_dir.c_str(), currframe);
    gpu_sys.WriteParticleFile(std::string(filename));

    // int steps_per_frame = 50000;
    //    int steps_per_frame = 2000; // how often to output positions
    int steps_per_frame = 100;  // how often to output positions

    double frame_step = 1e-4;  // how often to update the boundary

    std::cout << "frame step " << frame_step << " seconds" << std::endl;
    // std::cout << "boundary particles updated at every 0.1 sec" << std::endl;

    while (curr_time < time_end) {
        // clock_t start_time = std::clock();
        gpu_sys.AdvanceSimulation(frame_step);
        // clock_t end_time = std::clock();

        // std::cout << "cpu time: " << ((double)(end_time - start_time))/CLOCKS_PER_SEC << std::endl;
        updateBoundaryLayerPosition(drum_diameter / 2.0f - sphere_radius, boundary_idx, drum_omega, gpu_sys,
                                    frame_step);

        updateBoundaryLayerVelocity(drum_diameter / 2.0f - sphere_radius, boundary_idx, drum_omega, gpu_sys);

        if (currframe % steps_per_frame == 0) {
            sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), (int)(currframe / steps_per_frame) + 1);
            gpu_sys.WriteParticleFile(std::string(filename));

            std::cout << curr_time << std::endl;
        }

        curr_time += frame_step;
        currframe++;
    }

    return 0;
}
