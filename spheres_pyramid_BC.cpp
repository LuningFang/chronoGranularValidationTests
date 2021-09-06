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
// Pyramid test with top ball set up as a sphere BC condition
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/ChGpuData.h"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

int main(int argc, char* argv[]) {
    double mu_k = 0.2;
    double cor = 0.4;
    double youngs_modulus = 2.0e6;
    double poisson_ratio = 0.3;
    double step_size = 1e-5;
    double bottom_sphere_mass = 1.0;  // 1kg
    double sphere_radius = 0.15;      // diameter 0.3m
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double bottom_sphere_density = bottom_sphere_mass / sphere_volume;
    double time_settle = 0.5f;
    double time_roll = 2.0f;
    double time_end = time_settle + time_roll;
    double sphere_inertia = 0.4 * bottom_sphere_mass * sphere_radius * sphere_radius;

    double half_gap_ratio = 0.2;  // gap = 2*half_gap_ratio*ball_radius

    if (argc != 3) {
        fprintf(stderr, "need input for rolling fric coef and mass\n");
        return 0;
    }

    float mu_roll = atof(argv[1]);
    float top_sphere_mass = atof(argv[2]);

    double rolling_friction_coeffS2S = mu_roll;
    double rolling_friction_coeffS2W = mu_roll;

    double box_X = 10 * sphere_radius;
    double box_Y = 10 * sphere_radius;
    double box_Z = 10 * sphere_radius;

    double grav_X = 0.0f;
    double grav_Y = -9.81;
    double grav_Z = 0.0f;

    // Setup simulation
    ChSystemGpu gran_sys(sphere_radius, bottom_sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    gran_sys.SetVerbosity(CHGPU_VERBOSITY::QUIET);

    gran_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gran_sys.SetRollingCoeff_SPH2SPH(rolling_friction_coeffS2S);
    gran_sys.SetRollingCoeff_SPH2WALL(rolling_friction_coeffS2W);

    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);

    gran_sys.UseMaterialBasedModel(true);

    gran_sys.SetYoungModulus_SPH(youngs_modulus);
    gran_sys.SetYoungModulus_WALL(youngs_modulus);

    gran_sys.SetRestitution_SPH(cor);
    gran_sys.SetRestitution_WALL(cor);

    gran_sys.SetPoissonRatio_SPH(poisson_ratio);
    gran_sys.SetPoissonRatio_WALL(poisson_ratio);

    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gran_sys.SetStaticFrictionCoeff_SPH2SPH(mu_k);
    gran_sys.SetStaticFrictionCoeff_SPH2WALL(mu_k);

    ChVector<float> ground_plate_pos(0.0, 0.0, 0.0);
    ChVector<float> ground_plate_normal(0.0, 1.0, 0.0f);
    size_t ground_plate_id = gran_sys.CreateBCPlane(ground_plate_pos, ground_plate_normal, true);

    // sphere initial position and velocity
    ChVector<float> initial_position_left(-sphere_radius - half_gap_ratio * sphere_radius, sphere_radius + 0.01, 0.0f);
    ChVector<float> initial_position_right(sphere_radius + half_gap_ratio * sphere_radius, sphere_radius + 0.01, 0.0f);
    ChVector<float> initial_position_top(0.0f, 3 * sphere_radius, 0.0f);

    ChVector<float> initial_velo_left(0.0, -0.1, 0.0);
    ChVector<float> initial_velo_right(0.0, -0.1, 0.0);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);

    std::vector<ChVector<float>> body_points;
    body_points.push_back(initial_position_left);
    body_points.push_back(initial_position_right);

    std::vector<ChVector<float>> body_vels;
    body_vels.push_back(initial_velo_left);
    body_vels.push_back(initial_velo_right);

    gran_sys.SetParticles(body_points, body_vels);

    // create top sphere as BC type
    size_t top_sphere_id = gran_sys.CreateBCSphere(initial_position_top, sphere_radius, true, true, top_sphere_mass);

    gran_sys.SetFixedStepSize(step_size);

    gran_sys.SetBDFixed(true);
    //    gran_sys.SetRecordingContactInfo(true);

    gran_sys.Initialize();

    gran_sys.DisableBCbyID(top_sphere_id);
    float curr_time = 0;
    int fps = 1000;
    int currframe = 0;
    string output_dir = "pyramid";
    filesystem::create_directory(filesystem::path(output_dir));
    ChVector<float> pos_t;
    ChVector<float> velo_t;
    ChVector<float> angular_velo_t;

    ChVector<float> pos_l;
    ChVector<float> velo_l;
    ChVector<float> angular_velo_l;

    double KE;

    while (curr_time < time_settle) {
        gran_sys.AdvanceSimulation(step_size);

        curr_time += step_size;
        currframe++;

        pos_l = gran_sys.GetParticlePosition(0);
        velo_l = gran_sys.GetParticleVelocity(0);

        KE = 0.5 * bottom_sphere_mass * velo_l.Length2();

        if (velo_l.Length() < 1.0e-4 && curr_time > 0.4) {
            break;
        }
        pos_t = gran_sys.GetBCSpherePosition(top_sphere_id);
        velo_t = gran_sys.GetBCSphereVelocity(top_sphere_id);
    }

    double start_time = curr_time;
    double endTime = start_time + time_roll;
    double topBallPosition = std::sqrt(4 * sphere_radius * sphere_radius - pos_l.x() * pos_l.x()) + pos_l.y();
    ChVector<double> top_ball_pos(0.0f, topBallPosition, 0.0f);
    gran_sys.EnableBCbyID(top_sphere_id);
    gran_sys.SetBCSpherePosition(top_sphere_id, top_ball_pos);

    double gap;
    while (curr_time < endTime) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;
        currframe++;
        pos_l = gran_sys.GetParticlePosition(0);
        velo_l = gran_sys.GetParticleVelocity(0);
        angular_velo_l = gran_sys.GetParticleAngVelocity(0);
        ChVector<float> pos_r = gran_sys.GetParticlePosition(1);

        pos_t = gran_sys.GetBCSpherePosition(top_sphere_id);
        velo_t = gran_sys.GetBCSphereVelocity(top_sphere_id);
    }

    ChVector<float> pos = gran_sys.GetParticlePosition(0);
    ChVector<float> velo = gran_sys.GetParticleVelocity(0);
    ChVector<float> angular_velo = gran_sys.GetParticleAngVelocity(0);

    pos_t = gran_sys.GetBCSpherePosition(top_sphere_id);
    velo_t = gran_sys.GetBCSphereVelocity(top_sphere_id);

    printf("mu_r = %.2f, m_top = %.2f, bot_vx = %e, bot_omic = %e, top_y = %e, top_vy = %e\n", mu_roll, top_sphere_mass,
           velo.x(), angular_velo.z(), pos_t.y(), velo_t.y());

    return 0;
}
