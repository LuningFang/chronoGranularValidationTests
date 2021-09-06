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
// validation test: two spheres moving towards each other and collision happens
// parameters same as test/scm_contact/utest_SMCp_cor_normal
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "GpuDemoUtils.hpp"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_gpu/physics/ChSystemGpu.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

enum integrator { CENTERED_DIFFERENCE = 0, EXTENDED_TAYLOR = 1, FORWARD_EULER = 2, CHUNG = 3 };

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("need input for cor and integrator\n");
        return -1;
    }

    double cor = atof(argv[1]);

    integrator mySolver = (integrator)(std::atoi(argv[2]));

    double sphere_radius = 0.5;
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double sphere_mass = 1.0;
    double sphere_density = sphere_mass / sphere_volume;
    double youngs_modulus = 2.0e5f;
    double poisson_ratio = 0.3;
    double step_size = 3.0e-5;
    double time_end = 0.5;
    // original dt 1e-5, tend 0.3;

    // big domain dimension
    double box_X = 8 * sphere_radius;
    double box_Y = 8 * sphere_radius;
    double box_Z = 8 * sphere_radius;

    // Setup simulation
    ChSystemGpu gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    // create plane BC at the bottom of BD
    float plate_pos_z = (float)(-3.0f * sphere_radius);
    ChVector<float> plane_pos(0.0f, 0.0f, plate_pos_z);
    ChVector<float> plane_normal(0.0f, 0.0f, 1.0f);
    size_t plane_bc_id = gran_sys.CreateBCPlane(plane_pos, plane_normal, true);

    // assign initial condition for the sphere
    std::vector<ChVector<float>> body_point;
    double ball_pos_z = -1.001394;
    // body_point.push_back(ChVector<float>(-1.25 * sphere_radius, 0.0, plate_pos_z + sphere_radius));
    // body_point.push_back(ChVector<float>( 1.25 * sphere_radius, 0.0, plate_pos_z + sphere_radius));
    body_point.push_back(ChVector<float>(-1.25 * sphere_radius, 0.0, ball_pos_z));
    body_point.push_back(ChVector<float>(1.25 * sphere_radius, 0.0, ball_pos_z));
    std::vector<ChVector<float>> body_velo;
    body_velo.push_back(ChVector<float>(1.0, 0.0, 0.0));
    body_velo.push_back(ChVector<float>(-1.0, 0.0, 0.0));

    gran_sys.SetParticles(body_point, body_velo);

    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gran_sys.SetPsiFactors(psi_T, psi_L);

    // set normal force model
    gran_sys.SetYoungModulus_SPH(youngs_modulus);
    gran_sys.SetYoungModulus_WALL(youngs_modulus);
    gran_sys.SetRestitution_SPH(cor);
    gran_sys.SetRestitution_WALL(cor);
    gran_sys.SetPoissonRatio_SPH(poisson_ratio);
    gran_sys.SetPoissonRatio_WALL(poisson_ratio);

    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);

    // set gravity
    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -9.81f;

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    // set time integrator
    gran_sys.SetFixedStepSize(step_size);

    switch (mySolver) {
        case integrator::CENTERED_DIFFERENCE:
            gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
            break;
        case integrator::FORWARD_EULER:
            gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::FORWARD_EULER);
            break;
        case integrator::EXTENDED_TAYLOR:
            gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::EXTENDED_TAYLOR);
            break;
        case integrator::CHUNG:
            gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CHUNG);
            break;
    }

    gran_sys.SetVerbosity(CHGPU_VERBOSITY::QUIET);
    gran_sys.SetBDFixed(true);
    gran_sys.Initialize();

    float curr_time = 0;
    ChVector<float> pos_L;
    ChVector<float> velo_L;
    ChVector<float> pos_R;
    ChVector<float> velo_R;

    while (curr_time < time_end) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;

        pos_L = gran_sys.GetParticlePosition(0);
        velo_L = gran_sys.GetParticleVelocity(0);

        pos_R = gran_sys.GetParticlePosition(1);
        velo_R = gran_sys.GetParticleVelocity(1);
    }

    printf("cor = %f, integrator: %d, ending velo, %e, err: %e\n", cor, mySolver, velo_R.x(),
           std::abs(cor - velo_R.x()));

    return 0;
}
