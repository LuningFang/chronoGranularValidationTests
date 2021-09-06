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
// Rolling ball on the ground
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
    double mu_k = 0.3;
    double cor = 0.0;
    double youngs_modulus = 2.0e5;
    double poisson_ratio = 0.3;
    double step_size = 3e-5;
    double sphere_mass = 1.0;    // 1kg
    double sphere_radius = 0.5;  // diameter 1m
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double sphere_density = sphere_mass / sphere_volume;
    double time_end = 2;
    double time_settle = 1.0f;
    double time_roll = 1.0f;
    double sphere_inertia = 0.4 * sphere_mass * sphere_radius * sphere_radius;

    float mu_roll = 0.2;
    float v_init_mag = 1;  // initial velo 1m/s

    double rolling_friction_coeffS2S = mu_roll;
    double rolling_friction_coeffS2W = mu_roll;

    double box_X = 10 * sphere_radius;
    double box_Y = 10 * sphere_radius;
    double box_Z = 10 * sphere_radius;

    double grav_X = 0.0f;
    double grav_Y = -9.81;
    double grav_Z = 0.0f;

    // Setup simulation
    ChSystemGpu gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

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
    ChVector<float> initial_position(0.0, sphere_radius + 0.01, 0.0f);
    ChVector<float> initial_velo(0.0, -0.1, 0.0);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);

    std::vector<ChVector<float>> body_points;
    body_points.push_back(initial_position);

    std::vector<ChVector<float>> body_vels(1, initial_velo);
    gran_sys.SetParticles(body_points, body_vels);

    gran_sys.SetFixedStepSize(step_size);

    gran_sys.SetBDFixed(true);

    gran_sys.Initialize();

    float curr_time = 0;
    int fps = 1000;
    int currframe = 0;
    string output_dir = "rolling";
    filesystem::create_directory(filesystem::path(output_dir));
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> angular_velo;

    double KE;

    while (curr_time < time_settle) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;
        currframe++;

        pos = gran_sys.GetParticlePosition(0);
        velo = gran_sys.GetParticleVelocity(0);

        KE = 0.5 * sphere_mass * velo.Length2();

        if (KE < 1.0e-9) {
            std::cout << "[settling] KE falls below threshold after " << curr_time << " s \n";
            break;
        }

        if (currframe % fps == 0) {
            std::cout << "t: " << curr_time << ", velo: " << velo.y() << std::endl;
        }
    }

    double start_time = curr_time;
    double endTime = start_time + time_roll;
    ChVector<double> roll_velo(1.0f, 0.0f, 0.0f);
    gran_sys.SetParticleVelocity(0, roll_velo);

    while (curr_time < endTime) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;
        currframe++;
        pos = gran_sys.GetParticlePosition(0);
        velo = gran_sys.GetParticleVelocity(0);
        angular_velo = gran_sys.GetParticleAngVelocity(0);

        KE = 0.5 * sphere_mass * velo.Length2() + 0.5 * sphere_inertia * angular_velo.Length2();

        if (KE < 1.0e-9) {
            std::cout << "[rolling] KE falls below threshold after " << curr_time - start_time << " s \n";
            break;
        }

        if (currframe % 100 == 0) {
            // std::cout << "t: "  << curr_time << ", velo: " << velo.x() << ", omic: " << angular_velo.z() <<
            // std::endl;
            printf("%e, %e, %e\n", curr_time - start_time, velo.x(), angular_velo.z());
        }
    }

    std::cout << "ending angular velocity: " << angular_velo.x() << ", " << angular_velo.y() << ", " << angular_velo.z()
              << std::endl;

    return 0;
}
